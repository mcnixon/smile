# Basic tests to ensure that the model is running

import smile

print('Running Earth-like test case:')

# Earth-like model
# 1/3 iron, 2/3 silicates
r_earthlike = smile.get_radius(1.0,P0=1e4,T0=300.0,Pad=1e6,x_w=0.0,x_g=0.0)

print('Radius is '+str(r_earthlike)+' Earth radii (should be around 0.97)')

print('Running sub-Neptune test case (unmixed envelope):')

# Sub-Neptune model
# 5% differentiated envelope, half H2O, half H/He, Earth-like nucleus
r_subnep = smile.get_radius(10.0,P0=1e4,T0=500.0,Pad=1e6,x_w=0.025,x_g=0.025,profiles=False)

print('Radius is '+str(r_subnep)+' Earth radii (should be around 3)')

print('Running sub-Neptune test case (mixed envelope):')

# Sub-Neptune model
# 5% differentiated envelope, half H2O, half H/He, Earth-like nucleus
r_subnep = smile.get_radius(10.0,P0=1e4,T0=500.0,Pad=1e6,x_w=0.025,x_g=0.025,mixed=True,profiles=False)

print('Radius is '+str(r_subnep)+' Earth radii (should be around 2.57)')
