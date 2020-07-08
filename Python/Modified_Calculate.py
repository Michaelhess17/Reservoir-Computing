from Delay_Reservoir import DelayReservoir
import numpy as np

t = DelayReservoir()

class mod_Delay_Res(DelayReservoir):

	def calculate(self,u,m,bits,t,act,no_act_res = False):
		"""
		Calculate reservoir state over duration u

		Args:
			u: input data
			m: mask array
			bits: number of bit precision, np.inf for analog values
			noise: noise amplitude in input
			t: ratio of node interval to solver timestep
			act: activation function to be used for nonlinear node
			no_act_res: Return, in addition to regular solution, the base vn for each series

		Returns:
			M_x: matrix of reservoir history
		"""
		
		
		#Reshape input data with mask

		J = self.mask(u,m)
		cycles = J.shape[0]
			
		#Add extra layer to match indexes with M_x
		J = np.vstack((np.zeros((1,self.N)),J))
		J = J.flatten(order= 'C')
		J = J.reshape((1,(1+cycles)*self.N),order = 'F')
		M_x = np.zeros(J.shape)
		Mx_no_act = np.zeros(J.shape)         # Create container to store values of denominator 
	
		
		#Select activation function
		a = self.activationFunction(act)

		#Iteratively solve differential equation with Euler's Method  
		for i in range(1,(cycles+1)*self.N*t//1): 
			vn = M_x[0,i-1]-M_x[0,i-1]*self.theta/t
			arg = 0
			if act == "wright":         # In the case that we want wright, take out the x(t) term by redefining vn
				vn = M_x[0,i-1]

			arg += M_x[0,i-1-self.tau*t] 
			vn += self.eta*a(self.beta*arg+self.gamma*J[0,(i-1)//t]) * self.theta/t
			M_x[0,i] = vn
			
		#Reshape matrix
		M_x_new = np.zeros((1+cycles,self.N*t))

		# M_x_new[:,i*self.N:(i+1)*self.N] = \
		#     M_x[0,i].reshape(1+cycles,self.N)         # Before M_x[i].reshape(1+cycles,self.N)  

		M_x_new = M_x.reshape(1+cycles,self.N)

		if no_act_res:

			# Create status containers
			Node_calced = []            # Stores when the node was calculated
			Jt_store = []           # Stores the input at ^
			x_t_store = []
			x_tau_store = []           # Stores x(t-tau) at ^

			# Loop through and solve no_activation portions
			for i in range(1,(cycles+1)*self.N*t//1): 
				arg = Mx_no_act[0,i-1-self.tau*t]

				vn_no_act = Mx_no_act[0,i-1]-Mx_no_act[0,i-1]*self.theta/t
				vn_no_act += self.eta*(self.beta*arg+self.gamma*J[0,(i-1)//t]) * self.theta/t

				if vn_no_act > 10:
					Node_calced.append(len(Mx_no_act)+1)
					Jt_store.append(J[0,(i-1)//t])
					x_t_store.append(Mx_no_act[0,i-1])
					x_tau_store.append(arg)

				Mx_no_act[0,i] = vn_no_act  # Store the denominator values (without the "1+", this can be added in if needed for computation)

			# Reshape Matrix
			Mx_new_nAct= np.zeros((1+cycles, self.N*t))  
			Mx_new_nAct = Mx_no_act.reshape(1+cycles,self.N)

			return M_x_new[1:,0:self.N*t:t], Mx_new_nAct[1:,0:self.N*t:t]  


		#Remove first row of zeroes, select values at node spacing
		return M_x_new[1:,0:self.N*t:t]

