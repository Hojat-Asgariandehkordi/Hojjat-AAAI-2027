# Method overview (Mermaid preview)

```mermaid
flowchart LR
  A[CT patch + box prompt] --> B[Mask hole from boxes]
  B --> C[Healthy RF inpainting]
  C --> D[Residual |x - x̂|]
  D --> E[Threshold + refine R]
  E --> F[Anomaly mask ŷ]
```
