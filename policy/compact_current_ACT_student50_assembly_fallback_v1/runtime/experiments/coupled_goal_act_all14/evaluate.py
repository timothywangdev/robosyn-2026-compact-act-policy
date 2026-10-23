"""Fixed FIT/MONITOR diagnostics, including executed-prefix versus generated-tail."""
from collections import Counter
import numpy as np
import torch
from experiments.goal_latent_act_arm_sequence_full.fk import hand_errors
from .data import batch
from .objective import opening_error
from .units import ARM,GRIP


@torch.no_grad()
def evaluate(model,data,ids,stats,fk,tasks,control,shuffle=False):
    model.eval();names=('parent','candidate','ungated');by={t:{h:{n:Counter() for n in names} for h in (10,50)} for t in tasks}
    accept=Counter();mixed=Counter();all50=Counter();first10=Counter()
    for start in range(0,len(ids),128):
        selected=np.asarray(ids[start:start+128]);i=torch.as_tensor(selected,device=data['physical_base'].device)
        x=batch(data,selected,stats,tasks,shuffled=shuffle);r=model(**x,state_action_only=control)
        target=data['physical_actions'][i];valid=data['valid'][i]
        values=dict(parent=r['parent_physical_actions'],candidate=r['physical_actions'],ungated=r['ungated_physical_actions'])
        proposal=r['ungated_physical_actions'][... ,list(ARM)].reshape(-1,50,2,6)
        prefix_bounds=((proposal[:,:10]>=model.lower6)&(proposal[:,:10]<=model.upper6)).all(-1).all(1)
        for task_index,task in enumerate(tasks):
            m=data['task_index'][i]==task_index
            accept[task]+=int(r['accepted'][m].sum());mixed[task]+=int((r['accepted'][m].sum(-1)==1).sum())
            all50[task]+=int(r['within_URDF_limits'][m].sum());first10[task]+=int(prefix_bounds[m].sum())
        for horizon in (10,50):
            mask=valid.clone();mask[:,horizon:]=False
            for name,action in values.items():
                e=hand_errors(action,target,mask,fk,indices=(0,9) if horizon==10 else (0,9,31,49))
                grip=opening_error(action,target,mask)
                for task_index,task in enumerate(tasks):
                    m=data['task_index'][i]==task_index;n=int(m.sum())
                    by[task][horizon][name].update(rows=n,hand_rows=2*n,joint=float(e['joint'][m].sum()),
                        position=float(e['position'][m].sum()),rotation=float(e['rotation'][m].sum()),opening=float(grip[m].sum()))
    def finish(v):
        n=max(v['hand_rows'],1)
        return dict(rows=v['rows'],arm_MAE_rad=v['joint']/n,FK_position_L2_m=v['position']/n,
                    FK_orientation_geodesic_rad=v['rotation']/n,opening_MAE_m=v['opening']/n)
    aggregate={h:{name:Counter() for name in names} for h in (10,50)}
    for t in tasks:
        for h in (10,50):
            for n in names:aggregate[h][n].update(by[t][h][n])
    return dict(rows=len(ids),metrics={str(h):{n:finish(v) for n,v in aggregate[h].items()} for h in (10,50)},
        per_task={t:dict(metrics={str(h):{n:finish(v) for n,v in by[t][h].items()} for h in (10,50)},
                        accepted_hand_chunks=accept[t],crosshand_mixed_accepted_chunks=mixed[t],
                        URDF_inside_hand_chunks_all50=all50[t],URDF_inside_hand_chunks_first10=first10[t]) for t in tasks},
        shuffled_roles_and_recomputed_goal=shuffle,selection=False,
        interpretation='Desired-command offline errors; not observed tracking/contact or robot success')
