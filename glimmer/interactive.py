from glimmer import set_field


def make_interactive(plotter, solver):

    plotter.add_key_event("u", update_scene)
    plotter.add_key_event("i", toggle_interactive)

    def update_scene():

        for obj, actor in actors:
            obj.transform(actor.GetMatrix())

            actor.SetUserTransform(None)
            actor.SetPosition(0, 0, 0)
            actor.SetOrientation(0, 0, 0)
            actor.SetScale(1, 1, 1)

        for obj in [*solver.probes, *solver.optics]:
            obj.clear_data()

        solver.solve()

        for i, (obj, actor) in enumerate(actors):
            if obj.dimensionality == 3:
                plotter.remove_actor(actor)
                new_actor = plotter.add_volume(
                    obj,
                    clim=clim,
                    cmap=cmap,
                    opacity_unit_distance=obj.length / np.linalg.norm(obj.dimensions),
                )
                actors[i] = (obj, new_actor)

        plotter.render()

        actors = [(obj, add_object(obj)) for obj in objects]

        # plotter.enable_parallel_projection()
        # plotter.show_grid()

        return plotter

    def toggle_interactive():

        if interactive:
            plotter.enable_trackball_style()
            interactive = False
        else:
            plotter.enable_trackball_actor_style()
            interactive = True


# def plot(self, scalars="|E|^2"):

# self.plotter = Plotter()

# objects = [*self.probes, *self.optics, self.source]

# all_scalars = np.concat([np.ravel(obj[scalars]) for obj in objects])

# self.clim = (np.nanmin(all_scalars), np.nanmax(all_scalars))

# def add_object(obj):
#     if obj.dimensionality == 3:
#         actor = self.plotter.add_volume(
#             obj,
#             clim=self.clim,
#             cmap=self.cmap,
#             opacity_unit_distance=obj.length / np.linalg.norm(obj.dimensions),
#             scalars=scalars,
#         )
#     else:
#         actor = self.plotter.add_mesh(
#             obj,
#             clim=self.clim,
#             cmap=self.cmap,
#             # opacity="linear",
#             scalars=scalars,
#         )

#     return actor
