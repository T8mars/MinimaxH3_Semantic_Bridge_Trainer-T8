"""Synthetic CPU/software export checks only; not H3 training or video evidence."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from tools.trainer import main
from wushu_bridge.bridge_model import BridgeConfig, build_bridge, load_bridge, save_bridge
from wushu_bridge.dataset import PairDataset, PairMeta
from wushu_bridge.inference_guidance import make_inference_guidance, write_inference_guidance
from wushu_bridge.trainer_engine import RunConfig, train_run


class InferenceGuidanceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory(prefix='synthetic_inference_')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_fixed_contract_overrides_starting_recipe_and_survives_reload(self):
        for alpha in (1., .12, .2):
            with self.subTest(alpha=alpha):
                contract = dict(schema=1, alpha_min=alpha, alpha_max=alpha, magnitude_match='per_token',
                    token_span='all', tail_ratio=1., chunk_tokens=0, auto_alpha=False, guard=False, allow_dim_mismatch=False)
                model = build_bridge(BridgeConfig(dim=8, hidden=8, layers=1, heads=2,
                    extra={'application_contract':contract, 'training_application':{'alpha_min':.05,'alpha_max':.35,'magnitude_mode':'none'}}))
                weight = self.root/f'synthetic-fixed-{alpha}.safetensors'
                save_bridge(str(weight),model)
                before = hashlib.sha256(weight.read_bytes()).hexdigest()
                result = write_inference_guidance(weight,load_bridge(str(weight)))
                self.assertEqual(result['status'],'fixed_contract')
                self.assertEqual(result['settings']['alpha'],alpha)
                self.assertEqual(result['settings']['magnitude_match'],'per_token')
                self.assertEqual(result['model_sha256'],before)
                self.assertEqual(hashlib.sha256(weight.read_bytes()).hexdigest(),before)
                self.assertIn('固定 alpha', (self.root/'模型使用说明.md').read_text(encoding='utf-8'))
                self.assertFalse(result['video_calibrated_by_this_export'])

    def test_normal_training_exports_guidance_without_claiming_calibration(self):
        rng = np.random.default_rng(9)
        ds = PairDataset(PairMeta(encoder='synthetic_not_h3'))
        for i in range(12):
            x = rng.normal(size=(3,8)).astype(np.float16)
            ds.add(x,x+.1,.2,.8,{'bad':f'synthetic bad {i}','good':f'synthetic good {i}'})
        cfg = RunConfig(hidden=8,heads=2,layers=1,bridge_epochs=1,batch_size=4,device='cpu',amp=False,magnitude_mode='none')
        run = self.root/'synthetic-run'
        result = train_run(ds,cfg,run,{'test':'synthetic_export'},stage='bridge')
        self.assertEqual(result['status'],'completed')
        guide = json.loads((run/'inference_settings.json').read_text(encoding='utf-8'))
        self.assertEqual(guide['status'],'unvalidated_starting_point')
        self.assertEqual(guide['settings']['alpha'],.12)
        self.assertEqual(guide['settings']['magnitude_match'],'none')
        self.assertEqual(guide['training_alpha_range'],[.05,.35])
        self.assertFalse(guide['video_calibrated_by_this_export'])
        model = load_bridge(str(run/'bridge.safetensors'))
        self.assertNotIn('application_contract',model.cfg.extra)
        self.assertEqual(make_inference_guidance(model)['settings'],guide['settings'])
        report = json.loads((run/'bridge_report.json').read_text(encoding='utf-8'))
        self.assertEqual(report['inference_guidance']['model_sha256'],guide['model_sha256'])
        self.assertIn('建议从 alpha', (run/'模型使用说明.md').read_text(encoding='utf-8'))
        self.assertFalse((run/'judge.safetensors').exists())

    def test_custom_range_has_matching_start_and_invalid_contract_fails(self):
        model = build_bridge(BridgeConfig(dim=8,hidden=8,layers=1,heads=2))
        guide = make_inference_guidance(model,{'alpha_min':.4,'alpha_max':.8,'magnitude_mode':'per_token'})
        self.assertAlmostEqual(guide['settings']['alpha'],.6)
        self.assertTrue(all(.4 <= a <= .8 for a in guide['trial_alphas']))
        with self.assertRaises(ValueError):
            make_inference_guidance(model,{'alpha_min':float('nan'),'alpha_max':.8})
        model.cfg.extra = {'application_contract':{'schema':1}}
        with self.assertRaises(ValueError):
            make_inference_guidance(model)

    def test_cli_regenerates_guide_for_existing_fixed_weight(self):
        contract = dict(schema=1,alpha_min=1.,alpha_max=1.,magnitude_match='per_token',token_span='all',
            tail_ratio=1.,chunk_tokens=0,auto_alpha=False,guard=False,allow_dim_mismatch=False)
        model = build_bridge(BridgeConfig(dim=8,hidden=8,layers=1,heads=2,extra={'application_contract':contract}))
        weight = self.root/'existing-synthetic.safetensors'
        save_bridge(str(weight),model)
        out = self.root/'usage'
        main(['usage','--bridge',str(weight),'--output-dir',str(out)])
        self.assertEqual(json.loads((out/'inference_settings.json').read_text())['settings']['alpha'],1.)


if __name__ == '__main__':
    unittest.main()
