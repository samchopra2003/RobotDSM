vm1_0, vf1_0, vs1_0, vus1_0 = -1,0,0,0
vm2_0, vf2_0, vs2_0, vus2_0 = -1,0,0,0
vm3_0, vf3_0, vs3_0, vus3_0 = -1,0,0,0
vm4_0, vf4_0, vs4_0, vus4_0 = -1,0,0,0
V = (vm1_0, vf1_0, vs1_0, vus1_0,
       vm2_0, vf2_0, vs2_0, vus2_0,
       vm3_0, vf3_0, vs3_0, vus3_0,
       vm4_0, vf4_0, vs4_0, vus4_0)


class AbstractState:
    def __init__(self, pattern_id=0, weights=[], V_state=V):
        self.pattern_id = pattern_id
        self.weights = weights
        self.V_state = V_state
    
    def set_weights(self, conv_weights):
        self.weights = conv_weights

    def get_weights(self):
        return self.weights

    def get_pattern_id(self):
        return self.pattern_id

    def set_V_state(self, V_state):
        self.V_state = V_state

    def get_V_state(self):
        return self.V_state
