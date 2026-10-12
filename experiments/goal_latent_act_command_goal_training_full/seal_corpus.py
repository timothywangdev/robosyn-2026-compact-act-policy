"""CPU-only final corpus proof; outer coordinator wait is explicitly unknown."""
import argparse
import json
from pathlib import Path
from . import data
from experiments.goal_latent_act_campaign.common import require,ref,write

def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    for k in ('campaign','campaign-sha256','output'):p.add_argument('--'+k,required=True)
    a=p.parse_args();require(data.sha(a.campaign)==a.campaign_sha256,'Exact terminal source coordinator receipt required')
    c=json.loads(Path(a.campaign).read_text())
    require(c['status']=='all63000_FIT_7000_MONITOR_frozen_ACT_records_published' and c['coverage_exact'] is True and (c['FIT'],c['MONITOR'])==(63000,7000),'Full actual final corpus required')
    owned=c['controller_identity'];require((owned['pid'],owned['birth_ticks'])==(155029,54941326),'Reviewed original coordinator required')
    require(data.identity(owned['pid']) is None and not data.group_members(owned['group']),'Actual original coordinator and group must drain')
    for path,digest in c['sources'].items():require(data.sha(path)==digest,'Original producer source changed')
    qualification=data.qualify_corpus(c)
    result=dict(schema='goal-latent-act-full-corpus-admission-v1',mode='complete_corpus_from_original_qualified_shards_outer_wait_unknown',
        campaign=ref(a.campaign),identity=dict(pid=owned['pid'],birth=owned['birth_ticks'],group=owned['group'],session=owned['session']),
        corpus_index=c['index'],source_counts={'FIT':63000,'MONITOR':7000},qualification=qualification,
        coordinator_original_outer_exit_code=None,coordinator_original_outer_wait_captured=False,
        observed_original_coordinator_absent=True,independently_observed_group_after_reap=[],
        scientific_evidence='Original shard0/reap/natural-drain receipts, exact source hashes and final record/payload identities; no inferred coordinator exit',GPU_queries=0,GPU_launches=0)
    require(not Path(a.output).exists(),'Fresh truthful final corpus proof required');write(a.output,result);print(result['mode'])

if __name__=='__main__':main()
