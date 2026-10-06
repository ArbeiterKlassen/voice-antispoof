# data

数据集不入库，本目录只保留 manifest、下载脚本与校验和。

## 现有数据

| 数据集 | 用途 | 规模 |
|--------|------|------|
| LibriSpeech dev-clean / test-clean / train-clean-100 | 真实语音池，克隆参考音 | 约 7 GB |
| ASVspoof2019 LA | 主基准，train / dev / eval 三划分 | 约 20 GB |
| ASVspoof2021 DF | 编解码与传输鲁棒性轨道 | 约 32 GB |

## manifest 约定

十五列格式，逐条记录来源与处理链。

```
utt_id,audio_path,label,source,clone_model,attack_type,attack_params,text,text_source,
prompt_utt_id,speaker_id,sample_rate,duration,split,sha256
```

`attack_params` 写成 mp3-64k / snr5 / rt60-0.4 / speed0.9 这类可直接解析的形式，
确实没有的字段写 na，不留空。

## 质量检查

每个数据集落盘后必须过三项。字节数与文件数比对。随机抽 5 条解码，打印采样率、时长、声道数。
sha256 写进 dataset_audit.csv。

下载完整性不能只看文件大小非零。DF 的分片与第三方权重都出现过静默截断，两次都是靠读
parquet footer 或加载校验才抓出来。

## 说话人泄漏

划分必须自检。train / dev / test 三个说话人集合的交集大小写进 split_check.txt，要求为 0。
DF 真实说话人与 LA-train、LA-dev 的交集同样要报。
