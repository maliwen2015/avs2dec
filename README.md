参考davs2和dav1d，实现的avs2解码器

# AVS2解码器性能测试结果

## 测试环境

- **CPU**: Intel Core i7-12700K (8P+4E, 12核/16线程)
- **测试序列**: 15gop.avs2 (704帧, 3840x2160, 10-bit)
- **编译模式**: Release, SSE4 SIMD加速
- **测试日期**: 2026-07-12

## 测试结果

| 线程数 | 并行模式 | 环路滤波 | FPS |
|:------:|:--------:|:--------:|:---------:|
| 1T | FRAME | on | 23.44 |
| 1T | FRAME | off | 27.29 |
| 1T | ROW | on | 23.70 |
| 1T | ROW | off | 27.52 |
| 4T | FRAME | on | 39.25 |
| 4T | FRAME | off | 49.39 |
| 4T | ROW | on | 52.61 |
| 4T | ROW | off | 63.42 |
| 8T | FRAME | on | 70.60 |
| 8T | FRAME | off | 83.13 |
| 8T | ROW | on | 76.52 |
| 8T | ROW | off | 94.34 |
| 16T | FRAME | on | 87.32 |
| 16T | FRAME | off | 99.70 |
| 16T | ROW | on | 79.13 |
| 16T | ROW | off | 101.12 |
