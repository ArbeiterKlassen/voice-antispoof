# 权重与检查点校验和（sha256）

生成于 2026-10-07。权重/检查点本体不入库（`*.pth` 全局忽略，见 `models/README.md`）；本表供复现核对。

## 官方权重

| 文件 | 字节 | sha256 | 已知答案 |
|---|---:|---|---|
| `code/aasist/models/weights/AASIST.pth` | 1281532 | `51d2d9cf0738172f61e2a384ec50a54a55363240f67c971ed55a92435bc1a1c0` | LA eval 0.8297% vs 0.83 |
| `code/aasist/models/weights/AASIST-L.pth` | 426428 | `814331d088032bb4c3fa61cc014789eadeed464209dd094ab3a2dd6ffbdce27a` | LA eval 0.991% vs 0.99 |

## 本项目训练产物（results/ 下）

| 检查点 | 字节 | sha256 | 说明 |
|---|---:|---|---|
| `results/det_aug/best.pth` | 1264244 | `21ffd787ac5f9f41c9668e8986527b2f9edf06f6909b010559351cee737e6fef` | 自建轨臂C/增广 |
| `results/det_aug/last.pth` | 1263988 | `3b5d3f4ca31012833f93b509576f697886c57eb40aad861843088c9d39bde7c7` | 自建轨臂C/增广 |
| `results/det_augr/best.pth` | 1264244 | `4118e551984778e168a48bdb5b38baf0e948f91c83eade370307d4f85f454338` | 自建轨臂D（随机增广） |
| `results/det_augr/last.pth` | 1263988 | `7a42af8cc3f00be9a867cd97ad7eb15fc50c69dc8991ae9149e513f896b75fcb` | 自建轨臂D（随机增广） |
| `results/det_holdout_formant/best.pth` | 1264180 | `045fa1aec210549bbb38459d2ee7894047ed4a831442026f5b38668f0cc1f3ca` | 训练产物 |
| `results/det_holdout_formant/last.pth` | 1263988 | `d001d6cf12f11a5289aacb928055a5d59a7168cbf7641bbe0e3c55ae830586ed` | 训练产物 |
| `results/det_signalproc20/best.pth` | 1264244 | `70b05b26c5f9f63293f1dc3480305cd01fb156c8ebf470130f180f4a3262736e` | 自建轨臂A（20ep，固定预算） |
| `results/det_signalproc20/last.pth` | 1263988 | `7179391f9411b71215a39c58cb4410287ba2ebd270c2d73d5e2eb623ebbd8352` | 自建轨臂A（20ep，固定预算） |
| `results/det_signalproc/best.pth` | 1264180 | `e5de4d7f78c3dcbecc3f49c83f9a4d305274d4f702eca0ee5e0dfda0bce626cc` | 自建轨臂B |
| `results/det_signalproc/last.pth` | 1263988 | `31d4a6dc06c26fffa408ed954fb8cdf4c6ed6476302328c0e1717d5eb87e5f5c` | 自建轨臂B |
| `results/la_arm_aug/best.pth` | 1264244 | `799a32dd0e7b6d2d966f1a43c4930a2af6ed3c487cff1e8109f5746a26eae736` | LA 增广臂（dev 最优备查） |
| `results/la_arm_aug/last.pth` | 1263988 | `6d6c122b1b4f11299b63b8b2188e8bf59c06db2fa23f1192ffa39cfb42fe7362` | LA 增广臂（固定预算主结果，§6.2） |
| `results/la_arm_ctl/best.pth` | 1264244 | `28b07fcd99aba17de9c20165ab4c68b53fb4312c602ac5f6d9d293c219ccae29` | LA 对照臂（dev 最优备查） |
| `results/la_arm_ctl/last.pth` | 1263988 | `09b8fd8d70774eb1e4d10073d61a99bf7d8759ef7f6af30db03a0d9d34cfbbeb` | LA 对照臂（固定预算主结果，§6.2） |
| `results/smoke_det/best.pth` | 1264180 | `cda056da0e9b908327d446f40b1efac29328211f5fdd483e3bec65594d7ed9e3` | 训练产物 |
| `results/smoke_det/last.pth` | 1263988 | `e6616e1e01cd0758e0bae8dbaaa36fb63aca9277280275247252d389419853f4` | 训练产物 |
| `results/ssl_base/best.pth` | 490342 | `df63d7b1b1a62c9067258cf1a3f8abab989cf69402c8510a43787a11b3c0ec7c` | 训练产物 |
| `results/ssl_base/last.pth` | 490086 | `d23b725792c766da524979492e2c0367e7c60cf428982f36a6d54b673fcd26d5` | 训练产物 |
| `results/ssl_smoke/best.pth` | 490342 | `da5d6f4678f5c66acbce61d929ae2863b410faccbe70d34232dd300ee8c8d681` | 训练产物 |
| `results/ssl_smoke/last.pth` | 490086 | `0fa97475ceacd5ed7c2ebe43dc9f7c65bcb06fa5cf14188cbc37f919749fafa5` | 训练产物 |
