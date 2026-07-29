import bpy


def handler(_scene, depsgraph):
    print("HANDLER")
    for update in depsgraph.updates:
        item = update.id
        print(
            "handler update",
            type(item).__name__,
            item.name,
            "geometry",
            update.is_updated_geometry,
            "transform",
            update.is_updated_transform,
            "shading",
            update.is_updated_shading,
        )


bpy.app.handlers.depsgraph_update_post.append(handler)


def dump(label, depsgraph):
    print("---", label)
    for update in depsgraph.updates:
        item = update.id
        print(
            "update",
            type(item).__name__,
            item.name,
            item.as_pointer(),
            "orig",
            getattr(item, "original", item).as_pointer(),
            "geometry",
            update.is_updated_geometry,
            "transform",
            update.is_updated_transform,
            "shading",
            update.is_updated_shading,
        )
    for item in depsgraph.ids:
        if isinstance(item, bpy.types.Material):
            original = item.original
            direct = original.evaluated_get(depsgraph)
            print(
                "material",
                item.name,
                "eval",
                item.as_pointer(),
                "is_eval",
                item.is_evaluated,
                "orig",
                original.as_pointer(),
                "direct",
                direct.as_pointer(),
            )


bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)
bpy.ops.mesh.primitive_plane_add()
obj = bpy.context.object
material = bpy.data.materials.new("ProbeMaterial")
material.use_nodes = True
obj.data.materials.append(material)

depsgraph = bpy.context.evaluated_depsgraph_get()
depsgraph.update()
dump("initial", depsgraph)

material.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.2
depsgraph.update()
dump("material edit", depsgraph)

obj.location.x = 2
depsgraph.update()
dump("transform", depsgraph)

material2 = bpy.data.materials.new("ProbeMaterial2")
material2.use_nodes = True
obj.data.materials.append(material2)
depsgraph.update()
dump("slot add", depsgraph)

obj.data.materials.pop()
depsgraph.update()
dump("slot remove", depsgraph)

group = bpy.data.node_groups.new("ProbeGroup", "ShaderNodeTree")
group.interface.new_socket(
    name="Color", in_out="OUTPUT", socket_type="NodeSocketColor"
)
group_output = group.nodes.new("NodeGroupOutput")
rgb = group.nodes.new("ShaderNodeRGB")
group.links.new(rgb.outputs["Color"], group_output.inputs["Color"])
group_node = material.node_tree.nodes.new("ShaderNodeGroup")
group_node.node_tree = group
material.node_tree.links.new(
    group_node.outputs["Color"],
    material.node_tree.nodes["Principled BSDF"].inputs["Base Color"],
)
depsgraph.update()
dump("group add", depsgraph)

rgb.outputs["Color"].default_value = (0.2, 0.3, 0.4, 1.0)
depsgraph.update()
dump("group edit", depsgraph)
