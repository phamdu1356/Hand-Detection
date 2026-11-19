"""3D closed-loop simulator: a jointed person filmed by a camera on the drone.

Each frame the person's 3D keypoints are projected through the drone camera, degraded by a
webcam-like noise model, and fed to the real Controller. The drone's motion changes what its
camera sees on the next frame, which a fixed laptop webcam cannot reproduce.
"""
