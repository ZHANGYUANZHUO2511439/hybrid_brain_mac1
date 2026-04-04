from matplotlib.tri import Triangulation
from pylab import *

def multiview(cortex,hemisphere_left,hemisphere_right,data, fig,suptitle='', figsize=(15, 10),
              CB_position=[0.27, 0.8, 0.5, 0.05], CB_orientation='horizontal', CB_fontsize=10,
              **kwds):
    cs = cortex
    vtx = cs.vertices
    tri = cs.triangles
    rm = cs.region_mapping
    x, y, z = vtx.T
    lh_tri = tri[np.unique(np.concatenate([ np.where(rm[tri] == i)[0] for i in hemisphere_left ]))]
    lh_vtx = vtx[np.concatenate([np.where(rm == i )[0] for i in hemisphere_left])]
    lh_x, lh_y, lh_z = lh_vtx.T
    lh_tx, lh_ty, lh_tz = vtx[lh_tri].mean(axis=1).T
    rh_tri = tri[np.unique(np.concatenate([ np.where(rm[tri] == i)[0] for i in hemisphere_right ]))]
    rh_vtx = vtx[np.concatenate([np.where(rm == i )[0] for i in hemisphere_right])]
    rh_x, rh_y, rh_z = rh_vtx.T
    rh_tx, rh_ty, rh_tz = vtx[rh_tri].mean(axis=1).T
    tx, ty, tz = vtx[tri].mean(axis=1).T

    views = {
        'lh-lateral': Triangulation(-x, z, lh_tri[argsort(lh_ty)[::-1]]),
        'lh-medial': Triangulation(x, z, lh_tri[argsort(lh_ty)]),
        'rh-medial': Triangulation(-x, z, rh_tri[argsort(rh_ty)[::-1]]),
        'rh-lateral': Triangulation(x, z, rh_tri[argsort(rh_ty)]),
        'both-superior': Triangulation(y, x, tri[argsort(tz)]),
    }


    def plotview(i, j, k, viewkey, z=None, zmin=None, zmax=None, zthresh=None, suptitle='', shaded=True, cmap=plt.cm.coolwarm, viewlabel=False):
        v = views[viewkey]
        ax = subplot(i, j, k)
        if z is None:
            z = rand(v.x.shape[0])
        if not viewlabel:
            axis('off')
        kwargs = {'shading': 'gouraud'} if shaded else {'edgecolors': 'k', 'linewidth': 0.1}
        if zthresh:
            z = z.copy() * (abs(z) > zthresh)
        tc = ax.tripcolor(v, z, cmap=cmap, **kwargs)
        tc.set_clim(vmin=zmin, vmax=zmax)
        ax.set_aspect('equal')
        if suptitle:
            ax.set_title(suptitle, fontsize=24)
        if viewlabel:
            xlabel(viewkey)
        return tc

    plotview(2, 3, 1, 'lh-lateral', data, **kwds)
    plotview(2, 3, 4, 'lh-medial', data, **kwds)
    plotview(2, 3, 3, 'rh-lateral', data, **kwds)
    plotview(2, 3, 6, 'rh-medial', data, **kwds)
    tc = plotview(1, 3, 2, 'both-superior', data, suptitle=suptitle, **kwds)
    subplots_adjust(left=0.0, right=CB_position[0], bottom=0.0, top=1.0, wspace=0, hspace=0)
    cax = fig.add_axes(CB_position)
    fig.colorbar(tc,cax=cax,orientation=CB_orientation)
    cax.tick_params(axis='y',labelsize=CB_fontsize)
    return cax

import matplotlib.animation as manimation
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def animation(cortex,hemisphere_left,hemisphere_right,data,begin,end,file_name='./test.gif'):

    my_dpi=100 # resolution
    # parameter fig
    text_fontsize = 20.0
    title_fontsize = 20.0
    fig = plt.figure(figsize=(900/my_dpi, 900/my_dpi), dpi=my_dpi) # size of windows

    def update_fig(i):
        print(i,end=' ' )
        fig.clf() # clear figure
        multiview(cortex,hemisphere_left,hemisphere_right,data[i],fig=fig,shaded=False)
        return [fig]
    anim = manimation.FuncAnimation(fig, update_fig,frames=np.arange(begin,end,1), blit=True)
    anim.save(file_name)
    plt.close('all')

def multiview_one(cortex,hemisphere_left,hemisphere_right,region,data, fig,suptitle='', figsize=(15, 10), **kwds):
    cs = cortex
    vtx = cs.vertices
    tri = cs.triangles
    rm = cs.region_mapping
    x, y, z = vtx.T
    lh_tri = tri[np.unique(np.concatenate([ np.where(rm[tri] == i)[0] for i in hemisphere_left ]))]
    lh_vtx = vtx[np.concatenate([np.where(rm == i )[0] for i in hemisphere_left])]
    lh_x, lh_y, lh_z = lh_vtx.T
    lh_tx, lh_ty, lh_tz = vtx[lh_tri].mean(axis=1).T
    rh_tri = tri[np.unique(np.concatenate([ np.where(rm[tri] == i)[0] for i in hemisphere_right ]))]
    rh_vtx = vtx[np.concatenate([np.where(rm == i )[0] for i in hemisphere_right])]
    rh_x, rh_y, rh_z = rh_vtx.T
    rh_tx, rh_ty, rh_tz = vtx[rh_tri].mean(axis=1).T
    tx, ty, tz = vtx[tri].mean(axis=1).T
    
    data = np.zeros_like(data)
    data[rm == region ] =  10.0

    views = {
        'lh-lateral': Triangulation(-x, z, lh_tri[argsort(lh_ty)[::-1]]),
        'lh-medial': Triangulation(x, z, lh_tri[argsort(lh_ty)]),
        'rh-medial': Triangulation(-x, z, rh_tri[argsort(rh_ty)[::-1]]),
        'rh-lateral': Triangulation(x, z, rh_tri[argsort(rh_ty)]),
        'both-superior': Triangulation(y, x, tri[argsort(tz)]),
    }


    def plotview(i, j, k, viewkey, z=None, zlim=None, zthresh=None, suptitle='', shaded=True, cmap=plt.cm.coolwarm, viewlabel=False):
        v = views[viewkey]
        ax = subplot(i, j, k)
        if z is None:
            z = rand(v.x.shape[0])
        if not viewlabel:
            axis('off')
        kwargs = {'shading': 'gouraud'} if shaded else {'edgecolors': 'k', 'linewidth': 0.1}
        if zthresh:
            z = z.copy() * (abs(z) > zthresh)
        tc = ax.tripcolor(v, z, cmap=cmap, **kwargs)
        if zlim:
            tc.set_clim(vmin=-zlim, vmax=zlim)
        ax.set_aspect('equal')
        if suptitle:
            ax.set_title(suptitle, fontsize=24)
        if viewlabel:
            xlabel(viewkey)

    plotview(2, 3, 1, 'lh-lateral', data, **kwds)
    plotview(2, 3, 4, 'lh-medial', data, **kwds)
    plotview(2, 3, 3, 'rh-lateral', data, **kwds)
    plotview(2, 3, 6, 'rh-medial', data, **kwds)
    plotview(1, 3, 2, 'both-superior', data, suptitle=suptitle, **kwds)
    subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0, wspace=0, hspace=0)


if __name__ == '__main__':
    animation()