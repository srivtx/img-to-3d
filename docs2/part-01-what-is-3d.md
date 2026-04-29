# Part 1: What is 3D? — The Absolute Beginner's Guide

## Before We Talk About AI, Let's Talk About Triangles

You probably know what a 3D model is. You've seen them in video games, movies, and maybe CAD software. But **how does a computer actually represent 3D?**

The answer is surprisingly simple: **triangles. Lots of them.**

---

## The Building Block: The Triangle

A triangle is the simplest shape that has an inside and an outside. Computers love triangles because:
- They're flat (easy to draw)
- They always fit on a plane (no weird curves)
- Any complex shape can be broken into triangles

### Vertices: The Corner Points

Every triangle has **3 corners**. Each corner is called a **vertex** (plural: vertices).

A vertex is just a point in 3D space with three numbers:
```
Vertex = (x, y, z)
```

- **x** = left/right
- **y** = up/down  
- **z** = forward/backward

Example:
```
v0 = (0, 0, 0)   # origin
v1 = (1, 0, 0)   # 1 unit to the right
v2 = (0, 1, 0)   # 1 unit up
```

These three vertices make one triangle.

### Faces: The Triangle Itself

A **face** is just a triangle defined by 3 vertex indices:
```
Face = [0, 1, 2]  # Use vertices v0, v1, v2
```

A 3D model is just:
- A list of vertices (points in space)
- A list of faces (which 3 vertices make each triangle)

That's it. **A 3D model is a list of points and a list of triangles.**

---

## Example: A Simple Cube

A cube has:
- 8 vertices (corners)
- 12 faces (each square side is 2 triangles, 6 sides × 2 = 12)

```
Vertices:
0: (-1, -1, -1)  # bottom-back-left
1: ( 1, -1, -1)  # bottom-back-right
2: ( 1,  1, -1)  # top-back-right
3: (-1,  1, -1)  # top-back-left
4: (-1, -1,  1)  # bottom-front-left
5: ( 1, -1,  1)  # bottom-front-right
6: ( 1,  1,  1)  # top-front-right
7: (-1,  1,  1)  # top-front-left

Faces (each is 3 vertex indices):
# Front face (2 triangles)
[4, 5, 6]
[4, 6, 7]

# Back face
[1, 0, 3]
[1, 3, 2]

# Top face
[3, 7, 6]
[3, 6, 2]

# Bottom face
[0, 1, 5]
[0, 5, 4]

# Right face
[1, 2, 6]
[1, 6, 5]

# Left face
[0, 4, 7]
[0, 7, 3]
```

**12 triangles = 1 cube.** Modern game characters have **millions** of triangles.

---

## File Formats: How We Save 3D Models

Computers need standard ways to save this vertex+face data. Here are the common formats:

### OBJ (Wavefront)
The simplest format. Human-readable text.
```
v 0 0 0       # vertex 0
v 1 0 0       # vertex 1
v 0 1 0       # vertex 2
f 1 2 3       # face using vertices 1, 2, 3
```
**Pros:** Simple, universal  
**Cons:** Separate file for textures, large for complex models

### PLY (Polygon File Format)
Like OBJ but with more metadata. Often used for point clouds.

### GLB/glTF (GL Transmission Format)
The modern web standard. **GLB** = everything in one binary file.
**Pros:** Single file, efficient, web-native (Three.js loads it directly)  
**Cons:** Binary (not human-readable)

### FBX (Filmbox)
Industry standard for animation/games. Supports bones, animation, materials.

### STL (STereoLithography)
Used for 3D printing. Just triangles, no colors or textures.

**Our project uses GLB** because:
1. It's one file (easy to download)
2. Browsers understand it natively
3. It supports colors and textures

---

## What Makes a 3D Model Look Good?

Just vertices and faces gives you a **wireframe** (like a skeleton). To make it look solid and pretty, you need:

### 1. Normals
A **normal** is a direction vector perpendicular to a triangle's surface. It tells the computer "which way is outward."

Normals are used for lighting. When light hits a surface, the angle between the light direction and the normal determines how bright that pixel is.

```
Normal = (0, 1, 0)  # pointing straight up
```

Without normals, everything looks flat and weird.

### 2. UV Coordinates (Texture Mapping)

A **texture** is a 2D image (like a photo) that gets "wrapped" around the 3D model.

**UV coordinates** map each vertex to a point on the 2D texture:
```
UV = (u, v)  # where u and v are between 0 and 1
```

Think of it like unfolding a cardboard box into a flat pattern. The UV map is that flat pattern.

### 3. Vertex Colors

Instead of a texture image, you can store a color directly on each vertex. The GPU interpolates colors between vertices.

**Pros:** Simple, no extra image files  
**Cons:** Lower detail than textures

### 4. Materials

A **material** defines how light interacts with the surface:
- **Diffuse color:** Base color
- **Specular:** Shininess (like plastic vs. metal)
- **Roughness:** How smooth/smudgy reflections are
- **Metalness:** Is it metal or not?

---

## From Image to 3D: The Core Problem

Now you understand what a 3D model is. But here's the magic question:

**How do you get from ONE 2D photograph to a 3D model?**

The photo is just **pixels** (color values). A 3D model needs **depth** (z-coordinates) for every point.

**The fundamental ambiguity:** A 2D photo has lost the depth information. Many different 3D shapes could produce the same 2D image.

### Example
Imagine a photo of a circle. It could be:
- A flat disk viewed straight-on
- A sphere viewed from any angle
- A cylinder viewed end-on
- A cone viewed from above

**The computer doesn't know which one is correct.** It has to guess based on:
1. **Training data:** The AI has seen millions of photos + 3D pairs
2. **Context:** Shadows, texture gradients, perspective
3. **Prior knowledge:** "Most circular things in photos are spheres or cylinders"

---

## Our Approach: Don't Guess Perfectly, Guess Fast

Traditional approaches try to get perfect 3D by:
- Taking 100 photos from every angle
- Running optimization for hours
- Iteratively refining

**We said: that's too slow.**

Instead:
1. **Coarse model:** Fast guess (~2 seconds). Might be wrong in details, but recognizable.
2. **Refinement:** Improve quality in background (~10 seconds).
3. **Progressive:** Show the coarse model immediately, upgrade later.

**This is the key insight:** Speed comes from accepting "good enough" first, then improving.

---

## Summary

| Concept | Simple Definition |
|---------|-------------------|
| **Vertex** | A point in 3D space (x, y, z) |
| **Face** | A triangle made of 3 vertices |
| **Mesh** | A collection of vertices + faces |
| **Normal** | Direction a surface faces (for lighting) |
| **UV** | Where a vertex maps on a 2D texture |
| **GLB** | A file format that packs mesh + textures together |
| **Coarse-to-fine** | Fast approximation first, refinement later |

---

**Next:** [Part 2: What is AI/ML?](part-02-what-is-ai.md) — We explain neural networks like you're five.
