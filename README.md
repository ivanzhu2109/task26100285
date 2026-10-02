# glbmeasure

GLB 静态几何计量库，供资产导入流程直接调用（无接口层）。输入本地 GLB
文件，返回场景三角面总数和以英寸表示的世界坐标轴对齐包围盒（AABB），
不保存任何派生数据。

## 使用

```python
from glbmeasure import measure

result = measure("/path/to/asset.glb")
print(result.triangle_count)  # 精确三角面数
print(result.bounds)          # ((min_x, min_y, min_z), (max_x, max_y, max_z)) 英寸
                              # 场景无三角面时为 None
```

依赖：Python 3.10+，numpy。

## 计量语义

- 遵循 glTF 2.0 的 GLB 语义；二进制资源取自内嵌 BIN 块。
- 文件指定默认 `scene` 时只计量该场景，否则取第一个场景；未引用的
  mesh 与其他场景不影响结果。
- 同一 mesh 被多个节点引用时，每个节点都是独立实例，分别贡献面数与
  空间范围；实例遵循节点的 `matrix` 或 TRS 语义，并受全部祖先变换
  影响（正确处理旋转、负缩放、非均匀缩放）。
- 只计量 `TRIANGLES`（含 mode 省略）、`TRIANGLE_STRIP`、`TRIANGLE_FAN`，
  按各自拓扑组成三角面；退化三角面仍计一面，其顶点参与范围计算。
- 支持合法 bufferView/accessor 偏移、交错布局 `byteStride`，以及
  POSITION 与索引上的稀疏 accessor（无基础 bufferView 时未覆盖元素
  按零解释）。
- 包围盒每个轴的极值都由实际参与三角面的世界坐标顶点取得；未被索引
  引用的顶点不扩大结果，旋转也不会产生额外留白。
- glTF 单位按米解释，输出除以 0.0254 转为英寸。

## 实现要点

- 整个文件以 `mmap` 只读映射，仅解析 JSON 块（≤2MiB）和实际被计量的
  bufferView；大型内嵌图片不会进入内存。
- 顶点变换按 256K 顶点分块进行，单次计量峰值内存增量有界（实测
  3M 顶点网格约 67MiB，限制 128MiB）。
- 包围盒累积使用 float64，世界坐标 10000 米内输出误差 ≤ 0.0001 英寸。

## 测试

```bash
python3 -m pytest tests/ -q
```

覆盖：旋转网格剔除未引用极值顶点、多实例不同祖先变换（含负缩放与
非均匀缩放、matrix 节点）、稀疏覆盖（有/无基础 bufferView）、交错
布局、稀疏索引、三种三角拓扑、退化三角面、场景选择与空结果、
大坐标精度，以及约 500MB 文件的内存与隔离性验证。
