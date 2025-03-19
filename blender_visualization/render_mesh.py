import blendertoolbox as bt
import bpy
import pickle
import os
import numpy as np
import math
import mathutils
import trimesh

meshPath = "PATH_TO_MESH.ojb"
planePath = "PATH_TO_PLANE.pkl"
pcPath = "PATH_TO_LEFTRIGHT.pkl"
outputPath = "PATH_TO_SAVE.png"
blendPath = "PATH_TO_SAVE.blend"
plane_id = 0
plane_global_scale = 0.8
render_plane = True
render_pc = True

## Set up the rendering
imgRes_x = 512  # Debug -> 512 / Final -> 1024
imgRes_y = 512  # Debug -> 512 / Final -> 1024
numSamples = 500 # Debug -> 100 / Final -> 500
exposure = 1.5
use_GPU = True
bt.blenderInit(imgRes_x, imgRes_y, numSamples, exposure, use_GPU)

## Good colors
Blue = (144.0/255, 210.0/255, 236.0/255, 1)
iglGreen = (153.0/255, 203.0/255, 67.0/255, 1)
coralRed = (250.0/255, 114.0/255, 104.0/255, 1)
caltechOrange = (255.0/255, 108.0/255, 12.0/255, 1)
royalBlue = (0/255, 35/255, 102/255, 1)
royalBlueAlpha = (0/255, 35/255, 102/255, 0.7)
royalYellow = (250.0/255, 218.0/255, 94.0/255, 1)
neutral = (0.4, 0.4, 0.4, 1)


if render_pc:
  #######################
  #### Point Cloud Begin
  #######################
  with open(pcPath, "rb") as f:
    allpcs = pickle.load(f)
  pc = allpcs[plane_id]
  pc = np.concatenate([pc["left"], pc["right"]], axis=0)

  location = (0, 0, 0.)
  rotation = (90, 0, 0)
  scale = (1, 1, 1)
  pc_obj = bt.readNumpyPoints(pc, location, rotation, scale)

  ptColor = bt.colorObj(royalBlueAlpha, 0.5, 1.3, 1.0, 0.0, 0.0)
  ptSize = 0.014
  bt.setMat_pointCloud(pc_obj, ptColor, ptSize)
  #######################
  #### Point Cloud End
  #######################

if render_plane:
  #######################
  #### Plane Begin
  #######################
  with open(planePath, "rb") as f:
    allplanes = pickle.load(f)
  plane = allplanes[plane_id]
  verts = plane.vertices
  verts *= plane_global_scale
  faces = plane.faces

  plane_mesh = bpy.data.meshes.new("sym_plane")
  plane_obj = bpy.data.objects.new("sym_plane", plane_mesh)

  plane_mesh.from_pydata(verts.tolist(), [], faces.tolist())
  plane_mesh.update(calc_edges=True)              # Update mesh with new data
  bpy.context.collection.objects.link(plane_obj)  # Link to scene
  bpy.context.view_layer.objects.active = plane_obj
  angle = (90. / 180. * np.pi, 0., 0.)
  plane_obj.rotation_euler = angle
  bpy.context.view_layer.update()

  meshColor = bt.colorObj(coralRed, 0.5, 1.5, 1.0, 0.0, 0.0)
  alpha = 0.45
  transmission = 0.
  bt.setMat_transparent(plane_obj, meshColor, alpha, transmission)
  #######################
  #### Plane End
  #######################

#######################
#### Mesh Begin
#######################
## read mesh
location = (0, 0, 0)
rotation = (90., 0, 0)
scale = (1., 1., 1.)
mesh = bt.readMesh(meshPath, location, rotation, scale)

meshC = bt.colorObj(neutral, 0.5, 1.0, 1.0, 0.0, 0.0)
subC = bt.colorObj(neutral, 0.5, 2.0, 1.0, 0.0, 1.0)
bt.setMat_ceramic(mesh, meshC, subC)
#######################
#### Mesh End
#######################

## set shading
bpy.ops.object.shade_smooth()

## set camera
camLocation = (1.5, 1.5, 0.8)
lookAtLocation = (0, 0, 0)
focalLength = 30
cam = bt.setCamera(camLocation, lookAtLocation, focalLength)
ttc = cam.constraints.new(type="TRACK_TO")
ttc.target = mesh
ttc.track_axis = "TRACK_NEGATIVE_Z"
ttc.up_axis = "UP_Y"

## set light
# sun light
lightAngle = (-203, 231, -518)
strength = 3
shadowSoftness = 0.3
sun = bt.setLight_sun(lightAngle, strength, shadowSoftness)
# ambient light
bt.setLight_ambient(color=(0.2, 0.2, 0.2, 1))

## set gray shadow to completely white with a threshold (optional but recommended)
bt.shadowThreshold(alphaThreshold = 0.05, interpolationMode = "CARDINAL")

## save blender file so that you can adjust parameters in the UI
bpy.ops.wm.save_mainfile(filepath=blendPath)

## save rendering
bt.renderImage(outputPath, cam)