import numpy as np
import mosek.fusion as mf

# Dummy inputs
x = np.array([0.5, 0.3, 0.2])
y = np.array([0.6, 0.4, 0.2])
xdot = np.array([0.1, 0.1, 0.1])
xddot = np.array([0.1, -0.05, 0.02])
f_measured = np.array([5.0, 2.0, -1.0])
Kv = np.diag([50, 50, 50])
K_prior = np.diag([300, 300, 300])
lambda_reg = 1.0

# Compute target residual
rhs = xddot + Kv @ xdot - f_measured
delta = y - x
rhs = rhs.reshape((3, 1))
delta = delta.reshape((3, 1))

def optimize_stiffness(delta, rhs, Khat, lambda_reg):
    M = mf.Model("stiffness_optimization")
    K = M.variable("K", [3, 3], mf.Domain.inPSDCone())

    pred_force = mf.Expr.mul(K, delta)
    residual = mf.Expr.sub(pred_force, rhs)
    residual_flat = mf.Expr.flatten(residual)
    residual_cost = mf.Expr.dot(residual_flat, residual_flat)

    Khat_param = M.parameter("Khat_param", [3, 3])
    Khat_param.setValue(Khat)
    diff = mf.Expr.sub(K, Khat_param)
    diff_flat = mf.Expr.flatten(diff)
    reg_cost = mf.Expr.dot(diff_flat, diff_flat)

    total_cost = mf.Expr.add(residual_cost, mf.Expr.mul(lambda_reg, reg_cost))
    M.objective("minimize_total", mf.ObjectiveSense.Minimize, total_cost)

    M.solve()
    return np.array(K.level()).reshape(3, 3)


K_opt = optimize_stiffness(delta, rhs, K_prior, lambda_reg)
print("Optimized stiffness matrix:\n", K_opt)