import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile
import json
import types
import numpy as np
from damage3d import supervised_direction_a as m

class NumericalTests(unittest.TestCase):
    def test_censor_limits(self):
        ix=np.array([0,0,1]); p=np.array([.9,.1,.8]); g=np.ones(3)
        a=m.aggregate(ix,p,g,2,.2,1e100,.5,1e-4)
        np.testing.assert_allclose(a['llr'],a['censored'],atol=1e-10)
        b=m.aggregate(ix,p,g,2,.2,1e-100,.5,1e-4)
        np.testing.assert_allclose(b['censored'],0,atol=1e-10)
    def test_jensen(self):
        a=m.aggregate(np.arange(3),np.array([.01,.2,.99]),np.ones(3),3,.2,1.,.5,1e-4)
        self.assertTrue(np.all(a['censored']>=a['pi_llr']-1e-12))
    def test_invalid(self):
        with self.assertRaises(ValueError): m.aggregate(np.array([0]),np.array([1.1]),np.ones(1),1,.2,1,.5,1e-4)
    def test_ap_ties(self):
        r=m.metrics(np.array([1,0]),np.array([.5,.5]),np.ones(2),.5)
        self.assertAlmostEqual(r['ap'],.5)
    def test_nll_stable(self):
        self.assertTrue(np.isfinite(m.nll(np.array([0,1]),np.array([0.,1.]),np.ones(2))))
    def test_unknown_not_negative(self):
        y=np.array([-1,0,1]); ok=y>=0
        self.assertEqual(int(ok.sum()),2)
    def fixture(self):
        C=len(m.DAMAGE_CLASSES); n=90
        d=dict(source_index=np.arange(n),split=np.repeat(['C_fit','C_cal','T'],30),
            region=np.repeat([f'r{i}' for i in range(9)],10),area=np.ones(n),
            labels=np.tile((np.arange(n)%2)[:,None],(1,C)))
        dtype=[('source_index','i8'),('camera','i8'),('gsd_frontal','f8'),('cos_theta','f8'),('map_scale','f8')]+[(f'p_{c}','f8') for c in m.DAMAGE_CLASSES]
        o=np.zeros(n*2,dtype=dtype); o['source_index']=np.repeat(np.arange(n),2); o['camera']=np.tile([0,1],n)
        o['gsd_frontal']=np.tile([.001,.01],n); o['cos_theta']=1; o['map_scale']=1
        for c in m.DAMAGE_CLASSES: o[f'p_{c}']=np.repeat(np.where(np.arange(n)%2,.8,.05),2)
        def iterator(run,ids):
            obs=o[np.isin(o['source_index'],ids)]
            for chunk in np.array_split(np.unique(obs['source_index']),3):
                yield obs[np.isin(obs['source_index'],chunk)]
        model=types.SimpleNamespace(rho=np.full(C,.2),p_clip=1e-4,
            view_gsd=lambda g,c,s:g*s/np.sqrt(c))
        q=dict(classes=[m.DAMAGE_CLASSES[0]],seed=0,sample_per_region=10,min_views=1,min_regions=3,
            max_cached_observations=1000,g50_grid=[.001,.01],beta_grid=[.5],alpha_grid=[.1,1.],threshold_grid=[.2,.5,.8])
        p={'classes':{c:{'rho':.2} for c in m.DAMAGE_CLASSES}}; prov={'fixture':True}
        return d,o,iterator,model,q,p,prov
    
    def test_sampling_weights(self):
        d,o,it,model,q,p,prov=self.fixture(); q['sample_per_region']=5
        rows,w=m.select_fit(d,q)
        self.assertEqual(len(rows),15); self.assertAlmostEqual(w.sum(),30)
        self.assertTrue(np.all(d['split'][rows]=='C_fit'))
        
    def test_pipeline_and_T_label_invariance(self):
        d,o,it,model,q,p,prov=self.fixture()
        def write(path,data): Path(path).write_text(json.dumps(data))
        context=lambda a:(q,d,p,model,prov)
        prob=lambda obs:np.column_stack([obs[f'p_{c}'] for c in m.DAMAGE_CLASSES])
        with tempfile.TemporaryDirectory() as td, patch.object(m,'context',side_effect=context),patch.object(m,'iter_observations',side_effect=it),patch.object(m,'probs_of',side_effect=prob),patch.object(m,'write_json',side_effect=write):
            td=Path(td); pre=td/'pre.json'; write(pre,{'passed':True,'provenance':prov,'config':q})
            a=types.SimpleNamespace(preflight=pre,out=td/'fit.json',run=td)
            m.fit(a); fitted=json.loads(a.out.read_text()); a.fit=a.out; a.out=td/'freeze.json'; m.freeze(a)
            frozen=json.loads(a.out.read_text()); a.frozen=a.out; a.out=td/'eval.json'; m.evaluate(a)
            report=json.loads(a.out.read_text()); c=q['classes'][0]
            self.assertEqual(report['results'][c]['supported_points'],30)
            self.assertEqual(set(report['results'][c]['methods']),set(m.METHODS))
            d['labels'][d['split']=='T']=1-d['labels'][d['split']=='T']
            a.out=td/'fit2.json'; m.fit(a); self.assertEqual(fitted,json.loads(a.out.read_text()))
            a.fit=a.out; a.out=td/'freeze2.json'; m.freeze(a)
            frozen2=json.loads(a.out.read_text()); frozen.pop('fit_sha256'); frozen2.pop('fit_sha256')
            self.assertEqual(frozen,frozen2)
    def test_batch_scoring_matches_direct(self):
        d,o,it,model,q,p,prov=self.fixture(); c=q['classes'][0]
        best={method:dict(g50=.003,beta=.5,alpha=.2) for method in m.METHODS}
        artifact={'results':{c:dict(rho=.2,methods=best)}}
        a=types.SimpleNamespace(run=Path('.'))
        prob=lambda obs:np.column_stack([obs[f'p_{name}'] for name in m.DAMAGE_CLASSES])
        with patch.object(m,'iter_observations',side_effect=it),patch.object(m,'probs_of',side_effect=prob):
            rows,counts,scores=m.score_split(a,d,model,q,artifact,'T')
        obs=o[o['source_index']>=60]; ids=obs['source_index']-60
        agg=m.aggregate(ids,obs[f'p_{c}'],obs['gsd_frontal'],30,.2,.003,.5,1e-4)
        for method in m.METHODS:
            np.testing.assert_allclose(scores[c][method],m.probability(agg,method,.2,.2))

if __name__=='__main__': unittest.main()
