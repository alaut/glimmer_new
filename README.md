# Glimmer 

```
Alex Laut
Bridge 12
May 30 25
```

## Physical Optics

Fields radiated from an aperture can be summaried with the Stratton-Chu equations

 
$$E = \oint_S   ((n \times E) \times \nabla G + i k \eta (n \times H) G + (n  \cdot E) \nabla G  ) dS$$

$$H = \oint ((n \times H) \times \nabla G - \frac{i k}{\eta} (n \times E) G + (n \cdot H) \nabla G ) dS$$

where

$$G = \frac{e^{i k R}}{4 \pi R} \quad R = |r-r'| \quad \nabla ' G = \frac{r}{R}G(\frac{1}{R} - jk)$$
 
Reflected fields are given by

$$E = 2(E\cdot n)n - E$$
$$H = H - 2(H\cdot n)n$$

## EFIE MoM RWG Solver
PEC scattering is described by

$$n \times (E - j \omega A - \nabla \phi) = 0$$

where

$$A = \mu \oint JG \quad \phi = \frac{1}{\epsilon}\oint\sigma G \qquad  \quad \nabla \cdot J = -j \omega \sigma $$

and

$$G = \frac{\exp(-i k R)}{4\pi R} \qquad R = |r-r'|$$
letting

$$\omega\epsilon = \frac{k}{\eta} \quad \omega \mu = \eta k$$

we yield the following simplified EFIE equation

$$\boxed{n \times E = n \times i k \eta \oint (JG + \frac{1}{k^2}\nabla'\cdot J \nabla G)dS'}$$

Presume we are dealing with arbitrary surfaces modeled by triangular patches

![Arbitrary surface modeled by triangular patches](trimesh.png)

we can then develop a basis function associated to each interior edge that vanishes everwhere on S except within the two triangles attached to that edge

![Triangle pair and geometrical parameters associated with interior edge](rwg.png)

The vector basis is defined as

$$f_n(r) = \begin{cases} \frac{l_n}{2A_n^+}(r-r^+), r \in T_n^+\\
\frac{l_n}{2A_n^-}(r-r_n^-), r \in T_n^-    
\end{cases}$$

The surface divergence is given by

$$\nabla \cdot f_n = \begin{cases} +\frac{l_n}{A_n^+}, r \in T_n^+\\ -\frac{l_n}{A_n^-}, r \in T_n^- \end{cases}$$

The currents are therefore approximated by summing current contributions over the edge basis n

$$J(r) \approx \sum_{n=1}^N I_nf_n(r)$$

The goal is then to solve the following

$$ZI = V$$

where

$$Z = l [\frac{j \omega}{2} (A^+\cdot \rho^+ + A^-\cdot \rho^-) + \phi^- - \phi^+]$$

where

$$V = \frac{l}{2}(E^+\cdot \rho^+ + E^-\cdot\rho^-)$$

$$A = \mu \oint f_n(r')G(R_m)dS'$$

$$R_m = |r_m^c\pm-r'$$

# References

1. S. Rao, D. Wilton, and A. Glisson, “Electromagnetic scattering by surfaces of arbitrary shape,” IEEE Trans. Antennas Propagat., vol. 30, no. 3, pp. 409–418, May 1982, doi: 10.1109/TAP.1982.1142818.
2. C. J. Bouwkamp, “Diffraction Theory,” Rep. Prog. Phys., vol. 17, no. 1, pp. 35–100, Jan. 1954, doi: 10.1088/0034-4885/17/1/302.
