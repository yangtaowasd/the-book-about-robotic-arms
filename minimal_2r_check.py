import numpy as np

q1, q2 = 0.0, np.pi / 2
l1 = l2 = 0.5
c1 = c2 = 0.25
m1 = m2 = 1.0
I1 = I2 = 1.0 / 48.0
g0 = 9.81

J = np.array([
    [-l1*np.sin(q1)-l2*np.sin(q1+q2),
     -l2*np.sin(q1+q2)],
    [ l1*np.cos(q1)+l2*np.cos(q1+q2),
      l2*np.cos(q1+q2)]
])
a = I1+I2+m1*c1**2+m2*(l1**2+c2**2)
b = m2*l1*c2
d = I2+m2*c2**2
M = np.array([
    [a+2*b*np.cos(q2), d+b*np.cos(q2)],
    [d+b*np.cos(q2), d]
])
g = g0*np.array([
    (m1*c1+m2*l1)*np.cos(q1)
    + m2*c2*np.cos(q1+q2),
    m2*c2*np.cos(q1+q2)
])

v = np.array([0.0, 0.1])
f = np.array([2.0, -3.0])
qd = np.linalg.solve(J, v)
tau_ext = J.T @ f
assert np.allclose(qd, [0.2, -0.2])
assert np.allclose(tau_ext, [-2.5, -1.0])
assert np.allclose(qd @ tau_ext, v @ f)
assert np.all(np.linalg.eigvalsh(M) > 0)
print(J, M, g, qd @ tau_ext, sep="\n")
