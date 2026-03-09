# NCCL Cross-Node RDMA Communication Analysis

本文档详细分析了NCCL跨机RDMA通信的完整流程，包括`ncclSend`和`ncclRecv`的API调用、Kernel执行、Proxy线程处理以及底层RDMA操作。

## 目录

1. [架构概览](#架构概览)
2. [ncclSend 完整流程](#ncclsend-完整流程)
3. [ncclRecv 完整流程](#ncclrecv-完整流程)
4. [RDMA传输层详解](#rdma传输层详解)
5. [关键数据结构](#关键数据结构)
6. [CTS握手协议](#cts握手协议)
7. [核心源文件索引](#核心源文件索引)

---

## 架构概览

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           用户层 (User API)                              │
│  ncclSend() / ncclRecv() / ncclAllReduce() ...                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                         任务调度层 (Enqueue)                              │
│  ncclEnqueueCheck() → p2pTaskAppend() / collEnqueue()                   │
│  任务加入队列，等待CUDA Kernel执行                                        │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                         Kernel执行层 (Device)                            │
│  RunWorkBatch → runSend/runRecv → Primitives.directSend/directRecv     │
│  GPU Kernel直接操作发送/接收缓冲区                                        │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                         Proxy服务层 (Host Thread)                        │
│  ncclProxyService() → ncclProxyProgress()                              │
│  管理网络连接、内存注册、进度跟踪                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                         网络传输层 (Transport)                           │
│  ncclNet->isend/irecv → ncclIbIsend/ncclIbIrecv                        │
│  RDMA操作: ibv_post_send / ibv_post_recv                               │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## ncclSend 完整流程

### 流程图

```
┌──────────────────────────────────────────────────────────────────────────┐
│ 发送端 (Sender)                                                          │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. API入口 (collectives.cc)                                             │
│     ncclSend(sendbuff, count, datatype, peer, comm, stream)             │
│         ↓                                                                │
│     ncclEnqueueCheck(&info)  // 创建ncclInfo，进入任务调度               │
│                                                                          │
│  2. 任务入队 (enqueue.cc)                                                │
│     p2pTaskAppend()                                                      │
│         ↓                                                                │
│     创建 ncclTaskP2p {func=ncclFuncSend, buff=sendbuff, root=peer}      │
│     加入 planner.peers[peer].sendQueue                                   │
│                                                                          │
│  3. Kernel执行 (sendrecv.h)                                              │
│     RunWorkBatch<ncclFuncSendRecv>                                       │
│         ↓                                                                │
│     runSend<ProtoSimple>(tid, tn, group, work)                          │
│         ↓                                                                │
│     Primitives.directSend(cursor, cursor, n)                            │
│         ↓                                                                │
│     写入 ncclSendMem.head 通知Proxy有新数据                              │
│                                                                          │
│  4. Proxy处理 (proxy.cc)                                                 │
│     ncclProxyService 线程轮询                                            │
│         ↓                                                                │
│     sendProxyProgress() 检测 head 变化                                   │
│         ↓                                                                │
│     ncclNet->isend() → ncclIbIsend()                                    │
│                                                                          │
│  5. RDMA发送 (net_ib/p2p.cc)                                            │
│     ncclIbIsend()                                                        │
│         ↓                                                                │
│     等待接收方的CTS (ctsFifo中包含remote_addr + rkey)                    │
│         ↓                                                                │
│     ncclIbMultiSend()                                                    │
│         ↓                                                                │
│     构造 ibv_send_wr:                                                    │
│       - opcode = IBV_WR_RDMA_WRITE_WITH_IMM                             │
│       - sge.addr = 本地GPU数据地址                                       │
│       - sge.lkey = 本地内存key                                          │
│       - wr.rdma.remote_addr = 远端接收缓冲区地址                         │
│       - wr.rdma.rkey = 远端内存key                                      │
│         ↓                                                                │
│     ibv_post_send(qp, wr) → 数据通过RDMA直接写入远端GPU内存              │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### 关键代码路径

```c
// 1. API入口 - src/collectives.cc:195-206
ncclResult_t ncclSend(const void* sendbuff, size_t count,
                      ncclDataType_t datatype, int peer,
                      ncclComm_t comm, cudaStream_t stream) {
  struct ncclInfo info = { ncclFuncSend, "Send",
    NULL, (void*)sendbuff, count, datatype, ncclSum, peer, comm, stream, 1, 1 };
  return ncclEnqueueCheck(&info);
}

// 2. Kernel执行 - src/device/sendrecv.h:16-42
template<typename Proto>
__device__ void runSend(int tid, int tn, int group, struct ncclDevWorkP2p* work) {
  size_t bytes = work->sendBytes;
  int chunkSize = useLargeChunk ? NCCL_MAX_NET_SIZE : u32fp8Decode(work->sendChunkSize_u32fp8);

  Primitives<T, RedOp, FanAsymmetric<0, 1>, 1, Proto, 1>
    prims(tid, tn, nullptr, &work->sendRank, work->sendAddr, nullptr, ...);

  size_t cursor = 0;
  do {
    int n = min(size_t(chunkSize), bytes-cursor);
    prims.directSend(cursor, cursor, n);  // 核心发送操作
    cursor += n;
  } while (cursor < bytes);
}

// 3. RDMA发送 - src/transport/net_ib/p2p.cc:173-258
ncclResult_t ncclIbIsend(void* sendComm, void* data, size_t size, ...) {
  struct ncclIbSendComm* comm = (struct ncclIbSendComm*)sendComm;

  // 等待接收方CTS (Clear To Send)
  int slot = comm->base.fifoHead % NET_IB_MAX_REQUESTS;
  volatile struct ncclIbSendFifo* slots = comm->ctsFifo[slot];
  if (slots[0].idx != idx) { *request = NULL; return ncclSuccess; }

  // 创建发送请求
  struct ncclIbRequest* req;
  NCCLCHECK(ncclIbGetRequest(&comm->base, &req));
  req->type = NCCL_NET_IB_REQ_SEND;
  req->send.data = data;
  req->send.size = size;

  // 存储lkey
  for (int i = 0; i < comm->base.vProps.ndevs; i++) {
    req->send.lkeys[i] = mhandleWrapper->mrs[i]->lkey;
  }

  // 执行RDMA发送
  NCCLCHECK(ncclIbMultiSend(comm, slot));
}

// 4. 构造RDMA工作请求 - src/transport/net_ib/p2p.cc:43-171
ncclResult_t ncclIbMultiSend(struct ncclIbSendComm* comm, int slot) {
  for (int r=0; r<nreqs; r++) {
    struct ibv_send_wr* wr = comm->wrs+r;
    struct ibv_sge* sge = comm->sges+r;

    sge->addr = (uintptr_t)reqs[r]->send.data;     // 本地GPU地址
    sge->lkey = reqs[r]->send.lkeys[devIndex];     // 本地key
    wr->opcode = IBV_WR_RDMA_WRITE;                // RDMA写
    wr->wr.rdma.remote_addr = slots[r].addr;       // 远端地址
    wr->wr.rdma.rkey = slots[r].rkeys[devIndex];   // 远端key
  }

  // 最后一个WR带立即数据通知接收方
  lastWr->opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
  lastWr->imm_data = htobe32(size);

  // 提交到网卡
  NCCLCHECK(wrap_ibv_post_send(qp->qp, comm->wrs, &bad_wr));
}
```

---

## ncclRecv 完整流程

### 流程图

```
┌──────────────────────────────────────────────────────────────────────────┐
│ 接收端 (Receiver)                                                        │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. API入口 (collectives.cc)                                             │
│     ncclRecv(recvbuff, count, datatype, peer, comm, stream)             │
│         ↓                                                                │
│     ncclEnqueueCheck(&info)  // 创建ncclInfo，进入任务调度               │
│                                                                          │
│  2. 任务入队 (enqueue.cc)                                                │
│     p2pTaskAppend()                                                      │
│         ↓                                                                │
│     创建 ncclTaskP2p {func=ncclFuncRecv, buff=recvbuff, root=peer}      │
│     加入 planner.peers[peer].recvQueue                                   │
│                                                                          │
│  3. Kernel执行 (sendrecv.h)                                              │
│     RunWorkBatch<ncclFuncSendRecv>                                       │
│         ↓                                                                │
│     runRecv<ProtoSimple>(tid, tn, group, work)                          │
│         ↓                                                                │
│     Primitives.directRecv(cursor, n)                                    │
│         ↓                                                                │
│     轮询 ncclRecvMem.tail 等待数据到达                                   │
│                                                                          │
│  4. Proxy处理 (proxy.cc + net.cc)                                        │
│     ncclProxyService 线程                                                │
│         ↓                                                                │
│     recvProxyProgress()                                                  │
│         ↓                                                                │
│     Phase 1: Post Receive                                                │
│       - ncclNet->irecv() → ncclIbIrecv()                                │
│       - ncclIbPostFifo() 发送CTS给发送方                                 │
│                                                                          │
│     Phase 2: Wait Completion                                             │
│       - ncclNet->test() 检查RDMA完成                                     │
│       - 收到数据后更新 connFifo[buffSlot].size = 实际大小                │
│                                                                          │
│     Phase 3: Flush (GDR)                                                 │
│       - 如果使用GDR，需要flush GPU缓存                                   │
│       - 更新 tail 通知Kernel数据已就绪                                   │
│                                                                          │
│  5. RDMA接收准备 (net_ib/p2p.cc)                                        │
│     ncclIbIrecv()                                                        │
│         ↓                                                                │
│     Post Receive WR (等待RDMA_WRITE_WITH_IMM)                           │
│         ↓                                                                │
│     ncclIbPostFifo()                                                     │
│         ↓                                                                │
│     RDMA_WRITE CTS到发送方:                                              │
│       - localElem[i].addr = 接收缓冲区GPU地址                            │
│       - localElem[i].rkeys[j] = rkey                                    │
│       - localElem[i].tag = 匹配标识                                     │
│                                                                          │
│  6. 完成处理                                                              │
│     ncclIbTest() 轮询CQ获取完成通知                                      │
│         ↓                                                                │
│     收到 IBV_WC_RECV_RDMA_WITH_IMM CQE                                   │
│         ↓                                                                │
│     imm_data 包含接收到的数据大小                                        │
│     数据已直接写入GPU内存                                                │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### 关键代码路径

```c
// 1. API入口 - src/collectives.cc:208-219
ncclResult_t ncclRecv(void* recvbuff, size_t count,
                      ncclDataType_t datatype, int peer,
                      ncclComm_t comm, cudaStream_t stream) {
  struct ncclInfo info = { ncclFuncRecv, "Recv",
    NULL, recvbuff, count, datatype, ncclSum, peer, comm, stream, 1, 1 };
  return ncclEnqueueCheck(&info);
}

// 2. Kernel执行 - src/device/sendrecv.h:44-71
template<typename Proto>
__device__ void runRecv(int tid, int tn, int group, struct ncclDevWorkP2p* work) {
  size_t bytes = work->recvBytes;
  int chunkSize = useLargeChunk ? NCCL_MAX_NET_SIZE : u32fp8Decode(work->recvChunkSize_u32fp8);

  Primitives<T, RedOp, FanAsymmetric<1, 0>, 1, Proto, 1>
    prims(tid, tn, &work->recvRank, nullptr, nullptr, work->recvAddr, ...);

  size_t cursor = 0;
  do {
    int n = min(size_t(chunkSize), bytes-cursor);
    prims.directRecv(cursor, n);  // 核心接收操作
    cursor += n;
  } while (cursor < bytes);
}

// 3. Proxy接收进度 - src/transport/net.cc:1441-1685
static ncclResult_t recvProxyProgress(struct ncclProxyState* proxyState,
                                       struct ncclProxyArgs* args) {
  // Phase 1: Post Receive请求
  for (int s=0; s<args->nsubs; s+=args->subs[s].groupSize) {
    if (sub->posted < sub->nsteps) {
      // 获取接收缓冲区地址
      ptrs[subCount] = localBuff + buffSlot * stepSize;
      sizes[subCount] = stepSize * args->sliceSteps;
      tags[subCount] = resources->tpRemoteRank;
      mhandles[subCount] = sub->recvMhandle;

      // 发起异步接收
      NCCLCHECK(proxyState->ncclNet->irecv(
        resources->netRecvComm, subCount, ptrs, sizes,
        tags, mhandles, phandles, requestPtr));

      sub->posted += args->sliceSteps;
    }
  }

  // Phase 2: 检查接收完成
  for (int s=0; s<args->nsubs; s+=args->subs[s].groupSize) {
    if (subGroup->posted > subGroup->received) {
      NCCLCHECK(proxyState->ncclNet->test(
        subGroup->requests[step%NCCL_STEPS], &done, sizes));

      if (done) {
        // 更新连接FIFO，通知Kernel数据已就绪
        connFifo[buffSlot].size = -1;
        sub->received += args->sliceSteps;

        // GDR Flush处理
        if (needFlush) { ... }
      }
    }
  }

  // Phase 3: Flush完成，更新tail
  if (subGroup->received > subGroup->transmitted) {
    NCCLCHECK(proxyState->ncclNet->test(request, &done, NULL));
    if (done) {
      // 更新tail指针，通知GPU Kernel
      resources->recvMem->tail = ...;
      sub->transmitted += args->sliceSteps;
    }
  }
}

// 4. RDMA接收 - src/transport/net_ib/p2p.cc:336-402
ncclResult_t ncclIbIrecv(void* recvComm, int n, void** data,
                         size_t* sizes, int* tags, void** mhandles,
                         void** phandles, void** request) {
  struct ncclIbRecvComm* comm = (struct ncclIbRecvComm*)recvComm;

  struct ncclIbRequest* req;
  NCCLCHECK(ncclIbGetRequest(&comm->base, &req));
  req->type = NCCL_NET_IB_REQ_RECV;
  req->nreqs = n;

  // Post Receive WR (等待RDMA_WRITE_WITH_IMM)
  struct ibv_recv_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = req - comm->base.reqs;
  wr.sg_list = NULL;  // 不需要SGE，数据由RDMA WRITE直接写入
  wr.num_sge = 0;

  for (int i = 0; i < nqps; i++) {
    NCCLCHECK(wrap_ibv_post_recv(qp->qp, &wr, &bad_wr));
  }

  // 发送CTS (Clear To Send) 给发送方
  NCCLCHECK(ncclIbPostFifo(comm, n, data, sizes, tags, mhandles, req));

  *request = req;
}

// 5. 发送CTS - src/transport/net_ib/p2p.cc:260-334
ncclResult_t ncclIbPostFifo(struct ncclIbRecvComm* comm, int n,
                            void** data, size_t* sizes, int* tags,
                            void** mhandles, struct ncclIbRequest* req) {
  int slot = comm->base.fifoHead % NET_IB_MAX_REQUESTS;
  struct ncclIbSendFifo* localElem = comm->remCtsFifo.elems[slot];

  for (int i=0; i<n; i++) {
    localElem[i].addr = (uint64_t)data[i];  // 接收缓冲区地址

    // 发送rkey给发送方
    struct ncclIbMrHandle* mhandleWrapper = (struct ncclIbMrHandle*) mhandles[i];
    for (int j = 0; j < comm->base.vProps.ndevs; j++)
      localElem[i].rkeys[j] = mhandleWrapper->mrs[j]->rkey;

    localElem[i].nreqs = n;
    localElem[i].size = sizes[i];
    localElem[i].tag = tags[i];
    localElem[i].idx = comm->base.fifoHead + 1;
  }

  // 通过RDMA_WRITE把CTS发送给发送方
  wr.wr.rdma.remote_addr = comm->remCtsFifo.addr + slot*...;
  wr.wr.rdma.rkey = comm->base.remDevs[ctsQp->remDevIdx].rkey;
  wr.sg_list[0].addr = (uint64_t)localElem;
  wr.sg_list[0].length = n*sizeof(struct ncclIbSendFifo);
  wr.opcode = IBV_WR_RDMA_WRITE;
  wr.send_flags = IBV_SEND_INLINE;

  NCCLCHECK(wrap_ibv_post_send(ctsQp->qp, &wr, &bad_wr));
}

// 6. 完成测试 - src/transport/net_ib/p2p.cc:610-650
ncclResult_t ncclIbTest(void* request, int* done, int* sizes) {
  struct ncclIbRequest *r = (struct ncclIbRequest*)request;

  // 轮询CQ获取完成事件
  for (int i = 0; i < NCCL_IB_MAX_DEVS_PER_NIC; i++) {
    if (r->events[i] == 0) continue;

    NCCLCHECK(wrap_ibv_poll_cq(r->devBases[i]->cq, 4, wcs, &wrDone));

    for (int w=0; w<wrDone; w++) {
      struct ibv_wc *wc = wcs+w;

      if (wc->opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
        // 从立即数据获取接收大小
        if (req->nreqs == 1) {
          req->recv.sizes[0] = be32toh(wc->imm_data);
        }
      }

      r->events[i]--;
    }
  }

  if (ncclIbRequestIsComplete(r)) {
    *done = 1;
    // 返回接收到的数据大小
    if (sizes && r->type == NCCL_NET_IB_REQ_RECV) {
      for (int i=0; i<r->nreqs; i++) sizes[i] = r->recv.sizes[i];
    }
  }
}
```

---

## RDMA传输层详解

### InfiniBand操作类型

| 操作 | 说明 | 使用场景 |
|------|------|----------|
| `IBV_WR_RDMA_WRITE` | 单向RDMA写 | 大数据传输，不含通知 |
| `IBV_WR_RDMA_WRITE_WITH_IMM` | RDMA写+立即数据 | 发送数据并通知接收方 |
| `IBV_WR_RDMA_READ` | 单向RDMA读 | Flush操作 |
| `IBV_WR_SEND` | 双向发送 | 控制消息 |
| `IBV_WR_SEND_WITH_IMM` | 发送+立即数据 | - |

### 关键Verbs调用

```c
// 1. 创建QP (Queue Pair)
ibv_create_qp(pd, &qp_init_attr);

// 2. 修改QP状态
ibv_modify_qp(qp, &attr,
  IBV_QP_STATE | IBV_QP_PORT | IBV_QP_PKEY_INDEX |
  IBV_QP_ACCESS_FLAGS | IBV_QP_PATH_MTU | ...);

// 3. 注册内存
ibv_reg_mr(pd, addr, length, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);

// 4. 发送操作
ibv_post_send(qp, &wr, &bad_wr);

// 5. 接收操作
ibv_post_recv(qp, &wr, &bad_wr);

// 6. 轮询完成
ibv_poll_cq(cq, num_entries, wc);
```

---

## 关键数据结构

### ncclIbSendComm (发送端)

```c
struct ncclIbSendComm {
  struct ncclIbNetCommBase base;          // 基础通信结构
  struct ncclIbSendFifo* ctsFifo[...];    // 接收方发来的CTS
  struct ncclIbRequest** fifoReqs[...];   // 发送请求队列
  struct ncclIbRemCmplsRecords remCmplsRecords; // 远端完成记录
  struct ibv_send_wr* wrs;                // 发送工作请求
  struct ibv_sge* sges;                   // Scatter/Gather元素
  bool ar;                                // 自适应路由标志
};
```

### ncclIbRecvComm (接收端)

```c
struct ncclIbRecvComm {
  struct ncclIbNetCommBase base;          // 基础通信结构
  struct ncclIbRemCtsFifo remCtsFifo;     // 发送给发送方的CTS缓冲区
  int** cmplsRecords;                     // 完成记录数组
  bool flushEnabled;                      // 是否启用flush
};
```

### ncclIbSendFifo (CTS结构)

```c
struct ncclIbSendFifo {
  uint64_t addr;                          // 接收缓冲区地址
  uint32_t rkeys[NCCL_IB_MAX_DEVS_PER_NIC]; // 远端内存key
  int size;                               // 缓冲区大小
  int tag;                                // 匹配标签
  uint64_t idx;                           // 序列号
  int nreqs;                              // 请求数量
};
```

### ncclRecvMem (GPU端接收内存)

```c
struct ncclRecvMem {
  uint64_t tail;                          // 尾指针 (GPU写，Proxy读)
  char pad1[...];
  struct ncclConnFifo connFifo[NCCL_STEPS]; // 连接FIFO
  int flush;                              // Flush标志
};
```

### ncclSendMem (GPU端发送内存)

```c
struct ncclSendMem {
  uint64_t head;                          // 头指针 (GPU写，Proxy读)
  char pad1[...];
  void* ptrExchange;                      // 指针交换
  uint64_t redOpArgExchange[2];           // 规约操作参数
  char pad2[...];
  int offsFifo[NCCL_STEPS];               // 偏移量FIFO
};
```

---

## CTS握手协议

CTS (Clear To Send) 是NCCL RDMA通信的核心握手协议：

```
时间轴:    接收端                              发送端
           │                                    │
    T1     │ ncclRecv() 被调用                  │
           │    ↓                               │
           │ ncclIbIrecv()                      │
           │    ↓                               │
           │ Post Recv WR                       │
           │ (等待RDMA_WRITE_WITH_IMM)          │
           │                                    │
    T2     │ ncclIbPostFifo()                  │
           │    ↓                               │
           │ 构造CTS:                           │
           │   - addr = 接收缓冲区GPU地址       │
           │   - rkey = 远端访问key            │
           │   - size = 缓冲区大小             │
           │    ↓                               │
           │ RDMA_WRITE CTS ──────────────────→│
           │                                    │
    T3     │                                    │ ncclIbIsend() 检测CTS
           │                                    │    ↓
           │                                    │ 读取CTS中的addr和rkey
           │                                    │    ↓
           │                                    │ ncclIbMultiSend()
           │                                    │    ↓
           │                                    │ 构造RDMA WRITE:
           │                                    │   - remote_addr = CTS.addr
           │                                    │   - rkey = CTS.rkey
           │                                    │    ↓
           │←─────── RDMA_WRITE_WITH_IMM ──────│
           │   (数据直接写入GPU内存)            │
           │   (imm_data = 数据大小)            │
           │                                    │
    T4     │ CQE到达 (IBV_WC_RECV_RDMA_WITH_IMM)│
           │    ↓                               │
           │ 从imm_data获取大小                 │
           │    ↓                               │
           │ 数据已在recvbuff中就绪             │
           │                                    │
```

### CTS的优势

1. **Zero-copy**: 数据直接从发送端GPU内存到接收端GPU内存
2. **接收方控制**: 接收方决定何时、何地接收数据
3. **流控**: 发送方必须等待CTS才能发送，避免缓冲区溢出
4. **内存安全**: rkey机制确保发送方只能写入指定区域

---

## 核心源文件索引

### API层
| 文件 | 说明 |
|------|------|
| `src/collectives.cc` | ncclSend/ncclRecv等API入口 |
| `src/nccl.h` | 公共API头文件 |

### 任务调度层
| 文件 | 说明 |
|------|------|
| `src/enqueue.cc` | 任务入队和调度 |
| `src/include/comm.h` | ncclComm结构定义 |

### Kernel层
| 文件 | 说明 |
|------|------|
| `src/device/sendrecv.h` | Send/Recv Kernel实现 |
| `src/device/primitives.h` | 基础通信原语 |
| `src/device/prims_simple.h` | Simple协议实现 |
| `src/device/prims_ll.h` | Low-Latency协议实现 |
| `src/device/prims_ll128.h` | LL128协议实现 |

### Proxy层
| 文件 | 说明 |
|------|------|
| `src/proxy.cc` | Proxy线程实现 |
| `src/include/proxy.h` | Proxy结构定义 |

### 网络传输层
| 文件 | 说明 |
|------|------|
| `src/transport/net.cc` | 通用网络传输 |
| `src/transport/net_ib/p2p.cc` | InfiniBand P2P通信 |
| `src/transport/net_ib/connect.cc` | IB连接建立 |
| `src/transport/net_ib/common.cc` | IB通用功能 |

### 内存管理
| 文件 | 说明 |
|------|------|
| `src/transport/net_ib/reg.cc` | 内存注册 |
| `src/transport/net_ib/gdr.cc` | GPU Direct RDMA |

---

## 性能优化要点

1. **多QP并行**: 大数据可分片到多个QP并行传输
2. **Inline发送**: 小数据使用IBV_SEND_INLINE减少延迟
3. **自适应路由**: 大数据先发RDMA_WRITE，再发0字节RDMA_WRITE_WITH_IMM
4. **批处理**: 多个接收请求合并处理
5. **Pipeline**: 发送、接收、Flush阶段流水线执行

---

## 调试环境变量

```bash
# 启用详细日志
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=NET,PROXY

# IB相关配置
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_0,mlx5_1
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TIMEOUT=20
export NCCL_IB_RETRY_CNT=7

# 性能调优
export NCCL_IB_QPS_PER_CONNECTION=1
export NCCL_IB_SPLIT_DATA_ON_QPS=0
export NCCL_IB_AR_THRESHOLD=8192
```

---

*文档版本: 1.0*
*最后更新: 2024*
*基于NCCL源码版本分析*
