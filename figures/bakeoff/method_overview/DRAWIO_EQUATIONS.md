# Method-overview equations for draw.io

Paste each block into a draw.io text shape with **Extras → Mathematical Typesetting** enabled (MathJax `$$...$$`).

These match the paper (`eq:path`, `eq:loss`, `eq:inject`, `eq:sampler`, `eq:residual`, `eq:update`).

## Training

**Noise**
```latex
$$\boldsymbol{\varepsilon}\sim\mathcal{N}(\mathbf{0},\mathbf{I})$$
```

**Path + velocity (`eq:path`) — can share a cell with noise via a line break**
```latex
$$\boldsymbol{\varepsilon}\sim\mathcal{N}(\mathbf{0},\mathbf{I})$$
$$\mathbf{x}_t=(1-s)\mathbf{x}_0+s\boldsymbol{\varepsilon},\ \mathbf{u}=\mathbf{x}_0-\boldsymbol{\varepsilon},\ s=t/T$$
```

**Velocity loss (`eq:loss`)**
```latex
$$\mathcal{L}(\theta)=\mathbb{E}\!\left[\left\|v_\theta(\mathbf{x}_t,t)-(\mathbf{x}_0-\boldsymbol{\varepsilon})\right\|_1\right]$$
```

## Inference

**Initial hole**
```latex
$$\mathbf{h}^{(0)}_z=\mathrm{AABB}_{2\mathrm{D}}(\mathbf{y}_z)\oplus p$$
```

**Inject (`eq:inject`)**
```latex
$$\mathbf{x}_t\leftarrow\mathbf{m}\odot\tilde{\mathbf{x}}_t(\mathbf{x}_0)+(1-\mathbf{m})\odot\mathbf{x}_t$$
```

**Euler step (`eq:sampler`)**
```latex
$$\mathbf{x}_t\leftarrow\mathbf{x}_t+v_\theta(\mathbf{x}_t,t)\,\Delta s,\ \Delta s=(t-t_{\mathrm{next}})/T$$
```

**Residual (`eq:residual`)**
```latex
$$\mathbf{d}^{(r)}=|\mathbf{x}_0-\hat{\mathbf{x}}^{(r)}|$$
```

**Mask update (`eq:update`)**
```latex
$$\mathbf{h}^{\prime}=\mathbb{1}\!\left[\mathbf{d}^{(r)}>\tau\right]\odot\mathrm{Dilate}(\mathbf{h}^{(r)};\rho),\ \hat{\mathbf{y}}=\mathbf{h}^{(R)}$$
```

In the `.drawio` XML, `>` inside attributes is stored as `&gt;`.
