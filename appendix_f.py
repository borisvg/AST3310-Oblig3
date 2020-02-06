### program with functions used ###
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
# visulaliser
import FVis3

class solver:

    def __init__(self):

        """
        variable definitions
        """

        self.T0    = 5778                                   # [K]
        self.P0    = 1.8E8                                  # [kg m^-1 s^-2]
        self.kb    = 1.38064852e-23                         # [m^2 kg s^-2 K-1]
        self.nabla = 0.4001                                 # slightly larger than 2/5
        self.mu    = 0.61
        self.mu0   = 4.*np.pi * 1.e-7                       # permeability of free space
        self.sigma = 5.67e-8                                # electrical conductivity (?)
        self.m_u   = 1.660539040e-27                        # [kg]
        self.G     = 6.67408e-11                            # [m^3 kg^-1 s^-2]
        self.M_sun = 1.989e30                               # [kg]
        self.R_sun = 6.95508e8                              # [m]
        self.g     = -self.G * self.M_sun / self.R_sun**2   # [m/s^2], negative orientation
        self.cfl   = .1                                     # Courant-Friedrichs-Lewy condition
        self.eta   = .0001
        self.gamma = 5./3

        # box setup
        x0 = 0
        self.x1 = 12e6     # [m]

        y0 = 0             # top of the box
        self.y1 = 4e6      # [m]

        self.nx = 300
        self.ny = 100

        self.dx = self.x1/(self.nx-1)
        self.dy = self.y1/(self.ny-1)

        self.x = np.linspace(x0, self.x1, self.nx)
        self.y = np.linspace(y0, self.y1, self.ny)

        self.update = True


    def initialise(self,perturbation=False):

        """
        initialise temperature, pressure, density and internal energy
        """

        self.T   = np.zeros([self.ny, self.nx])
        self.P   = np.zeros([self.ny, self.nx])
        self.rho = np.zeros([self.ny, self.nx])
        self.ux  = np.zeros([self.ny, self.nx])
        self.uy  = np.zeros([self.ny, self.nx])
        self.e   = np.zeros([self.ny, self.nx])

        self.ux[:,:] = 0
        self.uy[:,:] = 0

        # set values at the top of the box (bottom in sim)
        self.T[-1,:] = self.T0
        self.P[-1,:] = self.P0

        # append values from the top down (bottom up in sim)
        for j in range(1,self.ny):
            self.T[-(j+1),:] = self.T0 - (self.mu * self.m_u * self.g * self.nabla * self.y[j]) / self.kb
            self.P[-(j+1),:] = self.P0 * (self.T[-(j+1),:]/self.T0)**(1./self.nabla)

        # add perturbation
        if perturbation:
            self.perturbation(self.T,self.T0, 0.2)     # const = 0.25 gives best results
            #self.perturbation(self.rho,self.rho[0,0], 0.1)

        self.rho = (self.mu * self.m_u * self.P) / (self.kb * self.T)
        #print(self.rho[:,1])
        self.e   = self.P / (self.gamma - 1.)

        # # set upper and lower values of the initial temperature
        self.T_top = self.T[-1,:]
        self.T_bot = self.T[0,:]

    def perturbation(self,func,func0,const):

        """
        create a gaussian perturbation in initial quantity
        f(x) = a * exp(-( (x-b)**2 + (y-c)**2 )/(2*d**2))
        """

        x,y = np.meshgrid(self.x, self.y)
        func += func0 * np.exp(-(((x - self.x1/2)**2 + (y - self.y1/2)**2)) / (1e6)**2) * const

    def soundspeed(self):

        """
        calculate speed of sound
        """

        # self.cs = (self.gamma * self.P / self.rho)**(1./2)
        self.cs = (self.gamma * (self.gamma - 1) * self.e)**(1./2)

    def timestep(self):

        """
        calculate timestep with relative quantities
        """

        max_rho = np.amax(abs(self.rho_dt/self.rho))
        max_e   = np.amax(abs(self.e_dt/self.e))
        # max_ux  = np.amax(abs(self.ux/self.dx))
        # max_uy  = np.amax(abs(self.uy/self.dy))

        # takes into account that velocity can be zero
        #print("dx",self.dx,np.amin(abs(self.ux)))
        #print(np.where(self.ux != 0.0,self.ux_dt/self.ux,self.ux_dt/self.dx))
 #       max_ux = np.amax(abs(np.where(self.ux != 0.0,self.ux_dt/self.ux,self.ux_dt/self.dx)))

        co = np.where(abs(self.ux) > 1.0/self.dx)
        scr=self.ux/self.dx

        scr[co] = self.ux_dt[co]/self.ux[co]
        max_ux = np.amax(scr)
        print("1: ",max_ux)
        if np.amax(abs(self.ux)) > 0.0:
            max_ux = np.amax(abs(self.ux_dt/self.ux))
        else:
            max_ux  = np.amax(abs(self.ux/self.dx))
        print("2: ",max_ux)
#        max_uy = np.amax(abs(np.where(self.uy != 0.0,self.uy_dt/self.uy,self.uy_dt/self.dy)))

        if np.amax(abs(self.uy)) > 0.0:
            max_uy = np.amax(abs(self.uy_dt/self.uy))
        else:
            max_uy  = np.amax(abs(self.uy/self.dy))


        self.max_num = np.array([max_rho, max_e, max_ux, max_uy]).max()

        #self.check_num = np.array([max_rho, max_e, max_ux, max_uy])

        if self.max_num > self.cfl:
            dt = self.cfl / self.max_num
        else:
            dt = self.cfl

        # if dt > self.cfl:
        #     dt = self.cfl

        return dt

    def boundary_conditions(self):

        """
        vertical boundary conditions for energy, density and velocity
        """

        constant_top = self.mu * self.m_u * self.g / (self.kb * self.T_top)
        constant_bot = self.mu * self.m_u * (-self.g) / (self.kb * self.T_bot)    # change sign of g at bottom

        self.e_top = (4.*self.e[-2,:] - self.e[-3,:]) / (3. - 2.*self.dy * constant_top)
        self.e_bot = (4.*self.e[1,:] - self.e[2,:]) / (3. - 2.*self.dy * constant_bot)

        self.rho_top = self.mu * self.m_u * (self.gamma - 1) / (self.kb * self.T_top) * self.e_top
        self.rho_bot = self.mu * self.m_u * (self.gamma - 1) / (self.kb * self.T_bot) * self.e_bot

        self.ux_top = (-self.ux[-3,:] + 4.*self.ux[-2,:])/3.
        self.ux_bot = (-self.ux[2,:] + 4.*self.ux[1,:])/3.

        # set boundary conditions
        self.e[-1,:]   = self.e[-1,:]-(self.e[-1,:]-self.e_top)/10.
        self.e[0,:]    = self.e[0 ,:]-(self.e[ 0,:]-self.e_bot)/10.
        self.rho[-1,:] = self.rho_top
        self.rho[0,:]  = self.rho_bot
        self.ux[-1,:]  = self.ux_top
        self.ux[0,:]   = self.ux_bot
        #self.uy[-1,:] = np.where(self.uy[-1,:] > 0.0, 0.0 , self.uy[-1,:])  #  = -self.uy[-2,:]
        #self.uy[ 0,:] = np.where(self.uy[0 ,:] < 0.0, 0.0 , self.uy[ 0,:])  #   = 0.
        self.uy[-1,:] = -self.uy[-2,:]
        self.uy[ 0,:] =  0.0

    def upwind_x(self,func,u):

        """
        includes left and right periodic boundaries
        excludes upper and lower boundaries
        """

        left_func = np.roll(func ,  1, axis=1)[1:-1]      # roll function to i-1 (since the array is "upside down")
        right_func = np.roll(func, -1, axis=1)[1:-1]    # roll function to i+1
        func2= func[1:-1]
        # upwind scheme for u>0
        return_func = (func2 - left_func)/self.dx

        # find where the velocity is negative
        y, x = np.where(u[1:-1,:]<0)
        if len(x) != 0:
            # upwind scheme for u[1:-1,:]<0
            return_func[y,x] = (right_func[y,x] - func2[y,x])/self.dx

        return return_func

    def upwind_y(self,func,u):

        """
        includes left and right periodic boundaries
        includes upper and lower boundaries, but not in roll
        """

        left_func  = np.roll(func,  1, axis=0)[1:-1, :]      # roll function to j-1, including boundaries
        right_func = np.roll(func, -1, axis=0)[1:-1, :]    # roll function to j+1, including boundaries

        # upwind scheme, excluding boundary calculation
        func2 = func[1:-1,:]
        return_func = (func2 - left_func)/self.dy

        y, x = np.where(u[1:-1,:]<0)
        if len(x) != 0:
            # upwind scheme, excluding boundary calculation
            return_func[y,x] = (right_func[y,x] - func2[y,x])/self.dy

        return return_func

    def central_x(self,func):

        """
        includes left and right periodic boundaries
        excludes upper and lower boundaries
        """

        right_func = np.roll(func, -1, axis=1)[1:-1, :]     # roll function to i+1
        left_func  = np.roll(func,  1, axis=1)[1:-1, :]      # roll function to i-1

        # central scheme
        return_func = (right_func - left_func) / (2.*self.dx)

        return return_func

    def central_y(self,func):

        """
        includes left and right periodic boundaries
        includes upper and lower boundaries, but not in roll
        """

        right_func = np.roll(func, -1, axis=0)[1:-1, :]      # roll function to j+1, including boundaries
        left_func  = np.roll(func,  1, axis=0)[1:-1, :]      # roll function to j-1, including boundaries

        # central scheme, excluding boundary calculation
        return_func = (right_func - left_func)/(2.*self.dy)

        return return_func

    def convective_flux(self):

        """
        calculate the convective fluc F_C
        """

        self.F_C = np.zeros(self.ny)

        for j in range(len(self.F_C)):
            self.F_C[j] = np.sum(self.e[j,:] * self.ux[j,:]) * self.dx

    def convective_velocity(self):

        """
        calculate the convective velocity
        """

        self.H_p = self.kb * self.T / (self.mu * self.m_u * (-self.g))
        self.c_p = 5./2 * self.kb / (self.mu * self.m_u)
        self.l_m = 1.

        # print(self.H_p, self.c_p)

        self.v = (self.F_C*self.H_p**(3./2)/(self.rho*self.c_p*self.T*self.g**(1./2))*(2/self.l_m)**2)**(1./3)

    def hydro_solver(self,order='first'):

        """
        hydrodynamic equations solver
        """

        # empty arrays
        self.rho_dt = np.zeros([self.ny,self.nx])
        self.e_dt   = np.zeros([self.ny,self.nx])
        self.ux_dt  = np.zeros([self.ny,self.nx])
        self.uy_dt  = np.zeros([self.ny,self.nx])

        # flux
        rhoux = self.rho*self.ux
        rhouy = self.rho*self.uy

        # central velocity
        cent_ux = self.central_x(self.ux)
        cent_uy = self.central_y(self.uy)

        # central pressure
        cent_px = self.central_x(self.P)
        cent_py = self.central_y(self.P)


        # density-velocity upwind
        upx_rho = self.upwind_x(self.rho,self.ux)
        upy_rho = self.upwind_y(self.rho,self.uy)

        # energy-velocity upwind
        upx_e = self.upwind_x(self.e,self.ux)
        upy_e = self.upwind_y(self.e,self.uy)

        # velocity upwind
        upx_uxx    = self.upwind_x(self.ux,self.ux)             # rhoux * dux_dx
        upy_uyx    = self.upwind_y(self.uy,self.ux)             # rhoux * duy_dy
        upx_rhouxx = self.upwind_x(rhoux,self.ux)               # ux * drhoux_dx
        upy_rhouxy = self.upwind_y(rhoux,self.uy)               # uy * drhoux_dy
        upx_uxy    = self.upwind_x(self.ux,self.uy)             # rhouy * dux_dx
        upy_uyy    = self.upwind_y(self.uy,self.uy)             # rhouy * duy_dy
        upy_rhouyy = self.upwind_y(rhouy,self.uy)               # uy * drhouy_dy
        upx_rhouyx = self.upwind_x(rhouy,self.ux)               # ux * drhouy_dx

        # central flux
        # upy_rhouxy = self.central_y(rhoux)
        # upx_rhouyx = self.central_x(rhouy)

        # gravity x density
        g_rho = self.g*self.rho[1:-1,:]

        # upx_uxx = self.central_x(self.ux)
        # upy_uyx = self.central_y(self.uy)
        # upx_uxy = self.central_x(self.ux)
        # upy_uyy = self.central_y(self.uy)

        # d/dt
        self.rho_dt[1:-1,:] = -self.rho[1:-1,:]*(cent_ux + cent_uy) - self.ux[1:-1,:]*upx_rho - self.uy[1:-1,:]*upy_rho

        self.e_dt[1:-1,:]   = -self.e[1:-1,:]*(cent_ux + cent_uy) - self.ux[1:-1,:]*upx_e - self.uy[1:-1,:]*upy_e \
                              - self.e[1:-1,:]*(self.gamma - 1)*(cent_ux + cent_uy)
#        self.e_dt[1, :]     = self.e_dt[1,:]-(self.e[1, :]-self.e[0, :])/0.001

        self.ux_dt[1:-1,:]  = (-rhoux[1:-1,:]*(upx_uxx + upy_uyx) - self.ux[1:-1,:]*upx_rhouxx - \
                               self.uy[1:-1,:]*upy_rhouxy - cent_px)/self.rho[1:-1,:]

        self.uy_dt[1:-1,:]  = (-rhouy[1:-1,:]*(upx_uxy + upy_uyy) - self.uy[1:-1,:]*upy_rhouyy - \
                               self.ux[1:-1,:]*upx_rhouyx - cent_py + g_rho )/self.rho[1:-1,:]

        # calculate timestep
        dt = self.timestep()


        # update variables
        if self.update == True :
            self.rho[:] = self.rho + self.rho_dt*dt
            self.e[:]   = self.e   + self.e_dt*dt
            self.ux[:]  = self.ux  + self.ux_dt*dt
            self.uy[:]  = self.uy  + self.uy_dt*dt

            scr = np.where(self.rho <= 0.0)
            if len(scr[0]) > 0 :
                print("Rho below zero at locations")
                for i in range(len(scr[0])) :
                    fx = scr[0][i]
                    fy = scr[1][i]
                    print("                (x,y) : ",i,fx,fy,self.rho[fx,fy])
#                    print("Terms: ",-self.rho[1:,:][fx-1,fy],cent_ux[fx-1,fy] ,cent_uy[fx-1,fy] )#,- self.ux[fx-1,fy]*upx_rho[fx-1,fy],self.uy[fx-1,fy]*upy_rho[fx-1,fy])
#                self.rho[scr[0],scr[1]] =100.
                    self.update = False

            scr = np.where(self.e <= 0.0)
            if len(scr[0]) > 0:
                print("E below zero at locations")
                for i in range(len(scr[0])):
                    fx = scr[0][i]
                    fy = scr[1][i]
                    print("                (x,y) : ", i, fx, fy, self.e[fx, fy])
                    self.update = False

        # initiate (vertical) boundary conditions for rho, e, ux and uy
        self.boundary_conditions()

        # compute temperature and pressure
        self.T[:] = (self.gamma - 1) * self.e * self.mu * self.m_u / (self.kb * self.rho)
        self.P[:] = (self.gamma - 1) * self.e

        # self.convective_flux()
        # self.convective_velocity()

        return dt

    def animate(self,seconds=150,variable='T'):

        """
        visualise simulation for specified variable (default is temperature)
        """
        print("variable",variable)
        vis.save_data(seconds, solve.hydro_solver, rho=solve.rho, u=solve.ux, w=solve.uy, T=solve.T, P=solve.P,\
                      sim_fps=0.1,folder=variable)
        vis.animate_2D(variable,showQuiver=True,quiverscale=3.,extent=[0,12,0,4,'Mm'],cmap='plasma',save=False)
        vis.delete_current_data()

    def animate_1D(self, array):

        """
        1D animation code
        """

        fig = plt.figure()
        ax = plt.axes(xlim=(0,4), ylim=(0,5e15))
        line, = ax.plot([], [], lw=2)
        ax.set_title('Time evolution of the total convective flux', fontsize=15)
        ax.set_xlabel(r'$y$ [Mm]', fontsize=13)
        ax.set_ylabel(r'$F_C$ [J m$^{-2}$ s$^{-1}$]', fontsize=13)

        def init():
            line.set_data([], [])
            return line,

        def animate(i):
            x = self.y/1e6        # [Mm]
            y = array[:, i]
            line.set_data(x, y)

            return line,

        anim = animation.FuncAnimation(fig, animate, init_func=init,
    								   frames=self.t_max, interval=200, blit=True)

        anim.save('total_pressure.mp4', fps=100, extra_args=['-vcodec', 'libx264'])
        plt.show()


    def sanity(self):

        """
        sanity tests
        """

        self.animate(seconds=120)


if __name__ == '__main__':

    # initialise visualiser and class
    vis = FVis3.FluidVisualiser()
    solve = solver()
    solve.initialise(perturbation=True)

    # run sanity check
    # solve.sanity()

    # animate results
    solve.animate(seconds=1000,variable='T')