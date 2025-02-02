# run Newton: python ms-2d.py
# run PD: python ms-2d.py -p True
import numpy as np
import taichi as ti
import taichi.math as tm
from scipy.sparse import bsr_matrix, csr_matrix, csc_matrix
from scipy.sparse.linalg import spsolve, factorized
from matplotlib.tri import Triangulation
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--pd', type=bool, default=False)
args = parser.parse_args()
enable_pd = args.pd

ti.init(arch=ti.cpu)

M, N = 16, 4
ks, mass, kp = 1e5, 1., 1e7
grav = ti.Vector([0., -9.8])
dx = 0.05

NV = (M + 1) * (N + 1)

X = np.zeros((NV, 2), dtype=np.float32)
for i in range(M + 1):
    for j in range(N + 1):
        X[i * (N + 1) + j] = [i * dx, j * dx]
X[:, 1] += 0.5
triangles = Triangulation(X[:, 0], X[:, 1])
T = triangles.triangles
# get all the edges in the triangulation
edges = set()
for i in range(triangles.triangles.shape[0]):
  for j in range(3):
    edge = (triangles.triangles[i, j], triangles.triangles[i, (j + 1) % 3])
    if edge[0] > edge[1]:
      edge = (edge[1], edge[0])
    edges.add(edge)
E = np.array(list(edges))

@ti.data_oriented
class MassSpring:
  def __init__(self, pos_np, ind_np, edge_np, pos_fixed_np, is_fixed_np, ks, mass, kp):
    self.pos_np = pos_np
    self.pos = ti.Vector.field(2, ti.f32, shape=(pos_np.shape[0],))
    self.pos.from_numpy(pos_np)
    self.ind_np = ind_np
    self.ind = ti.Vector.field(3, ti.i32, shape=(ind_np.shape[0],))
    self.ind.from_numpy(ind_np)
    self.edge_np = edge_np
    self.edge = ti.Vector.field(2, ti.i32, shape=(edge_np.shape[0],))
    self.edge.from_numpy(edge_np)
    self.rest_len = ti.field(ti.f32, shape=(edge_np.shape[0],))
    self.rest_len.from_numpy(np.linalg.norm(pos_np[edge_np[:, 0]] - pos_np[edge_np[:, 1]], axis=1))
    self.is_fixed = ti.field(ti.i32, shape=(pos_np.shape[0],))
    self.is_fixed.from_numpy(is_fixed_np)
    self.NV = self.pos.shape[0]
    self.NF = self.ind.shape[0]
    self.NE = self.edge.shape[0]
    self.ks = ks
    self.kp = kp
    self.n_iter = 0
    self.dx_norm = 0.

    self.pos_fixed = ti.Vector.field(2, ti.f32, shape=(self.NV,))
    self.pos_fixed.from_numpy(pos_fixed_np)
    self.pos_iner = ti.Vector.field(2, ti.f32, shape=(self.NV,))
    self.pos_last = ti.Vector.field(2, ti.f32, shape=(self.NV,))
    self.vel = ti.Vector.field(2, ti.f32, shape=(self.NV,))
    self.vel.fill(0.)
    self.mass = ti.field(ti.f32, shape=(self.NV,))
    self.mass.fill(mass)
    self.rhs = np.zeros(2 * self.NV, np.float32)
    
    self.quasi_static = True

    self.initHassian()
    
    self.Asp.data.fill(0.)
    self.fillAspData(self.Asp.data, self.e2off, self.v2off, dt)
    self.solve = factorized(csc_matrix(self.Asp))
  
  def buildGraph(self):
    nzs = []
    for i in range(self.ind_np.shape[0]):
      for a in range(3):
        for b in range(3):
          nzs.append((self.ind_np[i][a], self.ind_np[i][b]))
    nzs.sort()
    colptr = []
    col = []
    pre = (-1, -1)
    for i in range(len(nzs)):
      if nzs[i][0] != pre[0]:
        colptr.append(len(col))
      elif nzs[i][1] == pre[1]: continue
      col.append(nzs[i][1])
      pre = nzs[i]
    colptr.append(len(col))
    # c2off = []
    e2off = []
    v2off = []
    for edge in self.edge_np:
      i,j = edge
      for k in range(colptr[i], colptr[i + 1]):
        if col[k] == j:
          e2off.append(k)
          break
      for k in range(colptr[j], colptr[j + 1]):
        if col[k] == i:
          e2off.append(k)
          break
    for i in range(self.NV):
      for k in range(colptr[i], colptr[i + 1]):
        if col[k] == i:
          v2off.append(k)
          break
    return np.array(colptr), np.array(col), np.array(e2off), np.array(v2off)

  def initHassian(self):
    self.colptr, self.col, self.e2off, self.v2off = self.buildGraph()
    self.Asp = self.buildAsp()

  def buildAsp(self):
    data = np.zeros((len(self.col), 2, 2), np.float32)
    return bsr_matrix((data, self.col, self.colptr), blocksize=(2,2), shape=(2*self.NV, 2*self.NV))
  
  @ti.kernel
  def fillAspData(self, data: ti.types.ndarray(), e2off: ti.types.ndarray(), v2off: ti.types.ndarray(), dt: ti.f32):
    if self.quasi_static:
      for i in self.mass:
        diag = 0.
        if self.is_fixed[i]:
          diag += self.kp
        data[v2off[i], 0, 0] += diag
        data[v2off[i], 1, 1] += diag
    else:
      for i in self.mass:
        diag = self.mass[i] / dt / dt
        if self.is_fixed[i]:
          diag += self.kp
        data[v2off[i], 0, 0] += diag
        data[v2off[i], 1, 1] += diag
    for e in self.edge:
      edge = self.edge[e]
      th = self.ks * tm.eye(2)
      if not enable_pd:
        dir = self.pos[edge[1]] - self.pos[edge[0]]
        n = dir / tm.length(dir)
        th = self.ks * tm.eye(2) - self.ks * self.rest_len[e] / tm.length(dir) * (tm.eye(2) - n.outer_product(n))
      for x in ti.static(range(2)):
        for y in ti.static(range(2)):
          data[v2off[edge[0]], x, y] += th[x, y]
          data[v2off[edge[1]], x, y] += th[x, y]
          data[e2off[2 * e], x, y] -= th[x, y]
          data[e2off[2 * e + 1], x, y] -= th[x, y]

  @ti.kernel
  def computeEnergy(self, grav: ti.template(), dt: ti.f32) -> ti.f32:
    E = 0.
    # elastic
    for e in self.edge:
      edge = self.edge[e]
      dx = self.pos[edge[1]] - self.pos[edge[0]]
      E += self.ks / 2 * (tm.length(dx) - self.rest_len[e]) ** 2
    # iner
    if not self.quasi_static:
      for i in self.pos:
        dx = self.pos[i] - self.pos_iner[i]
        E += self.mass[i] / dt / dt * .5 * (dx[0] * dx[0] + dx[1] * dx[1])
    # fixed
    for i in self.pos:
      if self.is_fixed[i] > 0:
        dx = self.pos[i] - self.pos_fixed[i]
        E += .5 * self.kp * (dx[0] * dx[0] + dx[1] * dx[1])
    # gravity
    for i in self.pos:
      E += - self.mass[i] * grav[1] * self.pos[i][1] - self.mass[i] * grav[0] * self.pos[i][0]
    return E

  @ti.kernel
  def fillRhs(self, rhs: ti.types.ndarray(), grav: ti.template(), dt: ti.f32):
    for e in self.edge:
      edge = self.edge[e]
      dx = self.pos[edge[1]] - self.pos[edge[0]]
      d = tm.length(dx)
      f = self.ks * (d - self.rest_len[e]) / d
      for i in ti.static(range(2)):
        rhs[2 * edge[0] + i] += f * dx[i]
        rhs[2 * edge[1] + i] -= f * dx[i]
    # iner
    if not self.quasi_static:
      for i in self.pos:
        rhs[2 * i + 0] += self.mass[i] / dt / dt * (self.pos_iner[i][0] - self.pos[i][0])
        rhs[2 * i + 1] += self.mass[i] / dt / dt * (self.pos_iner[i][1] - self.pos[i][1])
    for i in self.pos:  
      if self.is_fixed[i] > 0:
        rhs[2 * i + 0] += self.kp * (self.pos_fixed[i][0] - self.pos[i][0])
        rhs[2 * i + 1] += self.kp * (self.pos_fixed[i][1] - self.pos[i][1])
    # gravity
    for i in self.pos:
      rhs[2 * i + 0] += self.mass[i] * grav[0]
      rhs[2 * i + 1] += self.mass[i] * grav[1]

  @ti.kernel
  def updateBasic(self, dt: ti.f32):
    for i in self.vel:
      self.pos_last[i] = self.pos[i]
      if not self.quasi_static:
        self.pos[i] += self.vel[i] * dt
      self.pos_iner[i] = self.pos[i]

  @ti.kernel
  def updatePosVel(self, dx:ti.types.ndarray(), alpha: ti.f32, dt: ti.f32):
    for i in self.pos:
      self.pos[i] += [dx[2 * i + 0] * alpha, dx[2 * i + 1] * alpha]
      self.vel[i] = (self.pos[i] - self.pos_last[i]) / dt

  def substepImplicit(self, grav, dt):
    self.updateBasic(dt)
    self.n_iter = 0
    self.rhs.fill(0.)
    self.fillRhs(self.rhs, grav, dt)
    dx = np.zeros(2 * self.NV, np.float32)
    if not enable_pd:
      self.Asp.data.fill(0.)
      self.fillAspData(self.Asp.data, self.e2off, self.v2off, dt)
      dx = spsolve(csr_matrix(self.Asp), self.rhs)
    else:
      dx = self.solve(self.rhs)
    self.updatePosVel(dx, 1., dt)
  
dt = 1e-2

is_fixed_np = np.zeros(NV, np.int32)
is_fixed_np[:N + 1] = 1
obj = MassSpring(X, T, E, X, is_fixed_np, ks, mass, kp)

def main():
  n_frame = 0
  run = False
  run_step = False
  window = ti.ui.Window("Mass Spring", (720, 720))
  canvas = window.get_canvas()
  gui = window.get_gui()
  canvas.set_background_color((.8, .8, .8))

  while window.running:
    if window.get_event(ti.ui.PRESS):
      if window.event.key == ti.ui.ESCAPE: break
      if window.event.key == 'r':
        n_frame = 0
      if window.event.key == 'p':
        run = not run
      if window.event.key == 's':
        run_step = True
    if run or run_step:
      obj.substepImplicit(grav, dt)
      run_step = False
      n_frame += 1
    
    with gui.sub_window("Control", .7, 0., .3, 0):
      if gui.button("Run"):
        run = not run
      if gui.button("Step"):
        run_step = True
      gui.text(f"frame: {n_frame}")
    
    canvas.circles(obj.pos, 5e-3)
    canvas.lines(obj.pos, 2e-3, obj.edge)
    window.show()

if __name__ == '__main__':
  main()

