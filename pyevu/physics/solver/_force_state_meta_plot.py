from __future__ import annotations
import os
from typing import TYPE_CHECKING
from ...util import require_dependencies
from ... import Vector2, Vector3
if TYPE_CHECKING:
    from ._state import ForceStateMeta

def plot(self: ForceStateMeta, save_dir: str=None):
    require_dependencies('matplotlib')
    import matplotlib.pyplot as plt
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    plist, vlist, alist, flist, tlist = self.get_pvaft_list()
    
    def _plot(
        tlist: list[float], flist: list[float],
        target_list: list[float], target_label: str,
        title: str="Force Impulse Simulation",
        save_path: str=None, show_force: bool=False
    ):
        ax = plt.subplot(111)
        for i in range(len(tlist)):
            t = tlist[i]; f = flist[i]; target = target_list[i]
            if show_force:
                plt.plot(
                    t, f, marker='+',
                    color='red'
                )
            plt.plot(
                t, target, marker='+',
                color='blue'
            )

        plt.title(title)
        plt.xlabel("time (in seconds)")
        plt.ylabel(target_label)

        if show_force:
            # Re-position Legend
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])    
            legend = plt.legend(['Applied Force', target_label], loc='upper left', bbox_to_anchor=(1, 1))
            legend.legendHandles[0].set_color('red')
            legend.legendHandles[1].set_color('blue')

        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.clf()
        plt.close('all')

    def _plot_attr(
        plist: list[float], vlist: list[float], alist: list[float],
        tlist: list[float], flist: list[float],
        attr_str: str=""
    ):
        for target_list, target_label, unit in [
            (plist, "Position", "m"),
            (vlist, "Velocity", "m/s"),
            (alist, "Acceleration", "m/s^2")
        ]:
            save_path = None
            if save_dir is not None:
                basename = target_label.lower()
                if attr_str != "":
                    basename = f"{basename}-{attr_str}"
                save_path = f"{save_dir}/{basename}.png"
            if attr_str != "":
                title = f"{target_label} {attr_str.upper()} (in {unit})"
            else:
                title = f"{target_label} {attr_str.upper()} (in {unit})"
            
            _plot(
                tlist=tlist, flist=flist,
                target_list=target_list,
                target_label=target_label,
                title=title,
                save_path=save_path
            )
    
    gen_type = self.get_generic_type()
    if gen_type is float:
        _plot_attr(plist=plist, vlist=vlist, alist=alist, tlist=tlist, flist=flist)
    elif gen_type in [Vector2, Vector3]:
        attr_str_list = ['x', 'y'] if gen_type is Vector2 else ['x', 'y', 'z']
        for attr_str in attr_str_list:
            _plist = [getattr(val, attr_str) for val in plist]
            _vlist = [getattr(val, attr_str) for val in vlist]
            _alist = [getattr(val, attr_str) for val in alist]
            _flist = [getattr(val, attr_str) for val in flist]
            _plot_attr(plist=_plist, vlist=_vlist, alist=_alist, tlist=tlist, flist=_flist, attr_str=attr_str)
    else:
        raise TypeError(f"Invalid type: {gen_type.__name__}")
