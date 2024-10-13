import mujoco


# class Collision_finger:


#     def __init__(self, vt_scene, target_pairs1, target_pairs2):
#         self.vt_scene = vt_scene
#         self.target_pairs1 = target_pairs1
#         self.target_pairs2 = target_pairs2
#         self.collision_l = 0
#         self.collision_r = 0

#     def check_collision(self, target_pairs):
#         if self.vt_scene.data is None or self.vt_scene.data.ncon <= 0:
#             return False
        
#         for i in range(self.vt_scene.data.ncon):
#             # Perform index check to ensure i is within range
#             if i >= len(self.vt_scene.data.contact):
#                 continue

#             contact = self.vt_scene.data.contact[i]
#             geom1_name = mujoco.mj_id2name(self.vt_scene.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
#             geom2_name = mujoco.mj_id2name(self.vt_scene.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
            
#             if (geom1_name, geom2_name) in target_pairs or (geom2_name, geom1_name) in target_pairs:
#                 return True

#         return False
    
#     def get_collisions(self):
#         if self.check_collision(self.target_pairs1):
#             self.collision_l = 1 
        
#         if self.check_collision(self.target_pairs2):
#             self.collision_r = 1 
        
#         return self.collision_l, self.collision_r
    

# class Collision_aim:

#     def __init__(self, vt_scene, target_pairs):
        
#         self.vt_scene = vt_scene
#         self.target_pairs = target_pairs
#         self.contact_aim = 0
    
#     def check_collision(self, target_pairs):
#         if self.vt_scene.data is None or self.vt_scene.data.ncon <= 0:
#             return False
        
#         for i in range(self.vt_scene.data.ncon):
#             contact = self.vt_scene.data.contact[i]
#             geom1_name = mujoco.mj_id2name(self.vt_scene.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
#             geom2_name = mujoco.mj_id2name(self.vt_scene.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
#             if (geom1_name, geom2_name) in target_pairs or (geom2_name, geom1_name) in target_pairs:
#                 return True

#         return False
    
#     def aim_resultant_force(self):
#         if self.check_collision(self.target_pairs):
#             self.contact_aim = 1 
#         return self.contact_aim
    



def check_collision(vt_scene, target_pairs):
        if vt_scene.data is None or vt_scene.data.ncon <= 0:
            return False
        
        for i in range(vt_scene.data.ncon):
            # Perform index check to ensure i is within range
            if i >= len(vt_scene.data.contact):
                continue

            contact = vt_scene.data.contact[i]
            geom1_name = mujoco.mj_id2name(vt_scene.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
            geom2_name = mujoco.mj_id2name(vt_scene.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
            
            if (geom1_name, geom2_name) in target_pairs or (geom2_name, geom1_name) in target_pairs:
                return True

        return False

def get_collisions(vt_scene,target_pairs1,target_pairs2):
    collision_l=0
    collision_r=0
    if check_collision(vt_scene,target_pairs1):
        collision_l = 1 
    
    if check_collision(vt_scene,target_pairs2):
        collision_r = 1 
    
    return collision_l, collision_r

def aim_resultant_force(vt_scene,target_pairs):
    contact_aim = 0
    if check_collision(vt_scene,target_pairs):
        contact_aim = 1 
    return contact_aim