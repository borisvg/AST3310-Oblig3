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
        self.dt = 0.01
        self.timesteps=0
        self.tottime=0.0

        # box setup
        x0 = 0
        self.x1 = 5e6     # [m]

        y0 = 0             # top of the box
        self.y1 = 2e6      # [m]

        self.nx = 300
        self.ny = 100+2

        self.dx = self.x1/(self.nx-1)
        self.dy = self.y1/(self.ny-1)

        self.x = np.linspace(x0, self.x1, self.nx)
        self.y = np.linspace(y0-self.dy, self.y1, self.ny)

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
        for j in range(1, self.ny):
            self.T[-(j+1),:] = self.T0 - (self.mu * self.m_u * self.g * self.nabla * self.y[j]) / self.kb
            self.P[-(j+1),:] = self.P0 * (self.T[-(j+1),:]/self.T0)**(1./self.nabla)

        # add perturbation
        if perturbation:
            self.perturbation(self.T,self.T, 0.01)     # const = 0.25 gives best results
            pt=(np.random.rand(self.ny,self.nx)-0.5)*2*0.001
            self.P[:,:] = self.P[:,:]*(1+pt)
            print(type(pt),pt.size,np.amax(pt),np.amin(pt))
            #self.perturbation(self.rho,self.rho[0,0], 0.1)

        self.rho = (self.mu * self.m_u * self.P) / (self.kb * self.T)
        #print(self.rho[:,1])
        self.e   = self.P / (self.gamma - 1.)

        # now make sure that even using the low order operators we have, the atmosphere is in equilibrium
        # the problem is only at boundaries.

        #for j in range(0, self.ny):
        #    print("%3i %8.3e %8.3e %8.3e %8.3e" % (j, self.T[j, 0], self.P[j, 0], self.rho[j, 0], self.e[j, 0]))

        # # set upper and lower values of the initial temperature
        self.T_top = self.T[-1:]
        self.T_bot = self.T[0 ,:]
        self.rho_top = self.rho[-1,:]
        self.rho_bot = self.rho[0,:]
        self.e_top = self.e[-1,:]
        self.e_bot = self.e[0,:]
        self.P_top = self.P[-1,:]
        self.P_bot = self.P[0,:]


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

        # NEW WAY
        scr=self.ux/self.dx
        co = np.where(abs(self.ux) > 1.0/self.dx)
        scr[co] = self.ux_dt[co]/self.ux[co]

        max_ux = np.amax((np.abs(self.ux)+np.abs(self.ux_dt)*self.dt)/self.dx)

#        if np.amax(abs(self.ux)) > 0.0:
#            max_ux = np.amax(abs(self.ux_dt/self.ux))
#        else:
#            max_ux  = np.amax(abs(self.ux/self.dx))

        # OLD WAY
#        max_ux = np.amax(abs(np.where(self.ux != 0.0, self.ux_dt / self.ux, self.ux_dt / self.dx)))
        #print("2: ",max_ux)

        #co = np.where(abs(self.uy) > 1.0/self.dy)
        #scr = self.uy/self.dy

        #scr[co] = self.uy_dt[co]/self.uy[co]
        #max_uy = np.amax(scr)

 #       if np.amax(abs(self.uy)) > 0.0:
 #           max_uy = np.amax(abs(self.uy_dt/self.uy))
 #       else:
 #           max_uy  = np.amax(abs(self.uy/self.dy))

        #max_uy = np.amax(abs(np.where(self.uy != 0.0,self.uy_dt/self.uy,self.uy_dt/self.dy)))

        max_uy = np.amax((np.abs(self.uy) + np.abs(self.uy_dt) * self.dt) / self.dy)

        self.max_num = np.array([max_rho, max_e]).max()

        #self.check_num = np.array([max_rho, max_e, max_ux, max_uy])
        dt2=0.0

        if self.max_num > self.cfl:
            dt2 = self.cfl / self.max_num
        else:
            dt2 = self.cfl

        self.dt=dt2
        self.timesteps = self.timesteps + 1
        self.tottime = self.tottime+self.dt
        # if dt > self.cfl:
        #     dt = self.cfl
        print("%4i %6.3f %6.3f %10.2e %10.2e %10.2e %10.2e" % (self.timesteps,self.dt,self.tottime,max_rho,max_e,max_ux,max_uy))

        return dt2

    def boundary_conditions(self):

        """
        vertical boundary conditions for energy, density and velocity
        """

        constant_top = self.mu * self.m_u * self.g / (self.kb * self.T_top)*0.001
        constant_bot = self.mu * self.m_u * (-self.g) / (self.kb * self.T_bot)*1000.   # change sign of g at bottom

#        self.e_top = (4.*self.e[-2,:] - self.e[-3,:]) / (3. - 2.*self.dy * constant_top)
#        self.e_bot = (4.*self.e[1,:] - self.e[2,:]) / (3. - 2.*self.dy * constant_bot)
#        print("len",len(constant_top),len(self.e_top),len(self.T_bot))
#        print("test",constant_top,self.e[-2,0],self.e[1,10],constant_bot)
#        self.rho_top = self.mu * self.m_u * (self.gamma - 1) / (self.kb * self.T_top) * self.e_top
#        self.rho_bot = self.mu * self.m_u * (self.gamma - 1) / (self.kb * self.T_bot) * self.e_bot
#        self.rho = (self.mu * self.m_u * self.P_top) / (self.kb * self.T_top)
        # print(self.rho[:,1])
#        self.e = self.P / (self.gamma - 1.)

        self.ux_top = (-self.ux[-3,:] + 4.*self.ux[-2,:])/3.
        self.ux_bot = (-self.ux[ 2,:] + 4.*self.ux[ 1,:])/3.

        # set boundary conditions
        # Forcing internal energy closer and closer to e_top and e_bot over 10 timesteps
#        self.e_dt[-1,:]   = self.e_dt[-1,:]-(self.rho[-1,:]/self.rho_top)**2*(self.e[-1,:]-self.e_top*0.8)/100.
#        self.e_dt[ 0,:]   = self.e_dt[ 0,:]-(self.e[ 0,:]-self.e_bot*1.01)/1000.


#        self.e_dt[-1, :] = - (self.rho[-1, :] / self.rho_top) * (self.e[-1, :] - self.e_top * 0.8) / 100.
        self.e[-1, :] = self.e_top*0.99
        self.e_dt[-2, :] = - (self.rho[-2,:]/self.rho_top)*(self.e[-2, :] - self.e[-1, :]) / 10000.
        self.e_dt[-1, :] = 0.0

        self.e_dt[1, :] = - (self.e[1, :] - self.e_bot) / 10000.
        self.e[0, :] = self.e_bot
        self.e_dt[0, :] = 0.0

        self.rho[-1,:]=self.rho_top
        self.rho[0,:] = self.rho_bot
        self.rho_dt[-1,:] = 0.0
        self.rho_dt[0,:] = 0.0


#        self.rho[-1,:] = self.rho_top
#        self.rho[0,:]  = self.rho_bot
#        self.ux[-1,:]  = self.ux_top
#        self.ux[0,:]   = self.ux_bot
#        self.uy[-1,:] = np.where(self.uy[-1,:] > 0.0, 0.0 , self.uy[-1,:])  #  = -self.uy[-2,:]
#        self.uy[ 0,:] = np.where(self.uy[0 ,:] < 0.0, 0.0 , self.uy[ 0,:])  #   = 0.
# Setting outflows to zero on upper and lower boundary

        co=np.where(self.uy[-1,:] > 0.0)
        scr=self.uy[-1,:]
        scr[co]=0.0
#        self.uy[-1,:] = scr
        self.uy[-1,:] = -self.uy[-2,:]
        self.uy_dt[-1,:] = 0.0
        self.uy[0,:] = -self.uy[1,:]
        self.uy_dt[0,:]=0.0

        self.ux[-1,:] = 0.0 #self.ux[-2,:]
        self.ux_dt[-1,:] = 0.0
        self.ux_dt[-2,:] = self.ux_dt[-2,:]*0.5
        self.ux[0,:] = 0.0
        self.ux_dt[0,:] = 0.0
        self.ux_dt[1,:] = self.ux_dt[1,:]*0.5

#        co=np.where(self.uy[0,:] < 0.0)
#        scr=self.uy[0,:]
#        scr[co]=0.0
#        self.uy[0,:]=scr
#        self.uy_dt[ 0,:] = -self.uy[ 0,:]/50.

    def upwind_x(self,func,u):

        """
        includes left and right periodic boundaries
        excludes upper and lower boundaries
        """

        left_func  = np.roll(func,  1, axis=1)   # roll function to i-1 (since the array is "upside down")
        right_func = np.roll(func, -1, axis=1)    # roll function to i+1
        func2= func

        # upwind scheme for u>0
        return_func = (func2 - left_func)/self.dx

        # find where the velocity is negative
        y, x = np.where(u < 0)
        if len(x) != 0:
            # upwind scheme for u[1:-1,:]<0
            return_func[y,x] = (right_func[y,x] - func2[y,x])/self.dx

        return return_func

    def upwind_y(self,func,u):

        """
        includes left and right periodic boundaries
        includes upper and lower boundaries, but not in roll
        """

        left_func  = np.roll(func,  1, axis=0)      # roll function to j-1, including boundaries
        right_func = np.roll(func, -1, axis=0)      # roll function to j+1, including boundaries

        # upwind scheme, excluding boundary calculation
        func2 = func
        return_func = (func2 - left_func)/self.dy

        y, x = np.where(u < 0)
        if len(x) != 0:
        #     # upwind scheme, excluding boundary calculation
            return_func[y,x] = (right_func[y,x] - func2[y,x])/self.dy

        # must convert to extrapolation scheme at boundary

#        return_func[ 0, :] = (right_func[0, :] - func2[0, :]) / self.dy
#        return_func[-1, :] = (func2[-1, :] - left_func[-1, :]) / self.dy

        #return_func[-1, :] = (right_func[-1, :] - func2[-1, :]) / self.dy
        #return_func[ 0, :] = (func2[0, :] - left_func[0, :]) / self.dy
        # x = np.where(u[-1,:] < 0)
        # if len(x) != 0:
        #     return_func[-1, :] = 0.0
        #
        # x=np.where(u[0,:] > 0)
        # if len(x) != 0:
        #     return_func[0,:] = 0.0

        return return_func

    def central_x(self,func):

        """
        includes left and right periodic boundaries
        excludes upper and lower boundaries
        """

        right_func = np.roll(func, -1, axis=1)      # roll function to i+1
        left_func  = np.roll(func,  1, axis=1)      # roll function to i-1

        # central scheme
        return_func = (right_func - left_func) / (2.*self.dx)

        return return_func

    def central_y(self,func):

        """
        includes left and right periodic boundaries
        includes upper and lower boundaries, but not in roll
        """

        right_func = np.roll(func, -1, axis=0)              # roll function to j+1, including boundaries
        left_func  = np.roll(func,  1, axis=0)              # roll function to j-1, including boundaries

        # central scheme, excluding boundary calculation
        return_func = (right_func - left_func)/(2.*self.dy)

        # central scheme becomes one-sided at boundaries - Need 3 points to maintain 2nd order
        #return_func[-1,:] = (0.5*func[-3,:]-2*func[-2,:]+1.5*func[-1,:])/self.dy
        #return_func[ 0,:] = (-1.5*func[0,:]+2*func[1,:]-0.5*func[2,:])/self.dy
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

        # initiate (vertical) boundary conditions for rho, e, ux and uy
        self.boundary_conditions()

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
        g_rho = self.g*self.rho

        # upx_uxx = self.central_x(self.ux)
        # upy_uyx = self.central_y(self.uy)
        # upx_uxy = self.central_x(self.ux)
        # upy_uyy = self.central_y(self.uy)

        # d/dt
        self.rho_dt =   self.rho_dt                     \
                        - self.rho*(cent_ux + cent_uy)  \
                        - self.ux*upx_rho               \
                        - self.uy*upy_rho

        self.e_dt   =   self.e_dt                       \
                        - self.e*(cent_ux + cent_uy)     \
                        - self.ux*upx_e                 \
                        - self.uy*upy_e                 \
                        - self.e*(self.gamma - 1)*(cent_ux + cent_uy)
#        self.e_dt[1, :]     = self.e_dt[1,:]-(self.e[1, :]-self.e[0, :])/0.001

# Trying with momentum

        px_dt = - rhoux * upx_uxx - self.ux * upx_rhouxx - cent_px \
                - rhoux * upy_uyy - self.uy * upy_rhouxy

        py_dt = - rhouy * upx_uxx - self.ux * upx_rhouyx \
                - rhouy * upy_uyx - self.uy * upy_rhouyy - cent_py

        self.ux_dt  =   (   -rhoux*(upx_uxx + upy_uyx)      \
                            -self.ux*upx_rhouxx             \
                            -self.uy*upy_rhouxy             \
                            -cent_px)                       \
                        /self.rho
        check = px_dt /self.rho -self.ux/self.rho * self.rho_dt-self.ux_dt

        print("CHECK: ",np.max(np.abs(check)))

        self.uy_dt  =   (   -rhouy*(upx_uxy + upy_uyy)      \
                            -self.uy*upy_rhouyy             \
                            -self.ux*upx_rhouyx             \
                            -cent_py + g_rho )              \
                        /self.rho


        self.boundary_conditions()

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
                      sim_fps=1,folder=variable)
        vis.animate_2D(variable,showQuiver=True,quiverscale=2.0,extent=[0,12,0,4,'Mm'],cmap='plasma',save=False,\
                       video_fps=46)
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
    								   frames=self.t_max, interval=0.001, blit=True)

        anim.save('total_pressure.mp4', fps=1000, extra_args=['-vcodec', 'libx264'])
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
    solve.animate(seconds=1200,variable='dP')

