# Basic tests to ensure that the model is running

import smile

print('Running Earth-like test case:')

# Earth-like model
# 1/3 iron, 2/3 silicates
r_earthlike = smile.get_radius(1.0,P0=1e4,T0=300.0,Pad=1e6,x_w=0.0,x_g=0.0)

print('Radius is '+str(round(r_earthlike,2))+' Earth radii (should be 0.97)')

print('Running sub-Neptune test case (unmixed envelope):')

# Sub-Neptune model
# 5% differentiated envelope, half H2O, half H/He, Earth-like nucleus
r_subnep = smile.get_radius(10.0,P0=1e4,T0=500.0,Pad=1e6,x_w=0.025,x_g=0.025,profiles=False)

print('Radius is '+str(round(r_subnep,2))+' Earth radii (should be 3.0)')

print('Running sub-Neptune test case (mixed envelope):')

# Sub-Neptune model
# 5% differentiated envelope, half H2O, half H/He, Earth-like nucleus
r_subnep = smile.get_radius(10.0,P0=1e4,T0=500.0,Pad=1e6,x_w=0.025,x_g=0.025,mixed=True,profiles=False)

print('Radius is '+str(round(r_subnep,2))+' Earth radii (should be 2.57)')
