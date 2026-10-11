"""CPU synthetic integration only; no real dataset, H3 encoding, GPU or quality claim."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import build_sentence_targets as builder
import fit_explicit_targets as fitter
import numpy as np
import torch
from wushu_bridge.bridge_model import load_bridge
from wushu_bridge.apply import apply_bridge_to_conditioning


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


class PublicIntegration(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="synthetic_explicit_bridge_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        self.data.mkdir()
        rng = np.random.default_rng(21)
        np.save(self.data / "x.npy", rng.normal(size=(3,8)).astype(np.float16))
        np.save(self.data / "y.npy", rng.normal(size=(4,8)).astype(np.float32))
        (self.data / "x.txt").write_text("abc", encoding="utf-8")
        (self.data / "y.txt").write_text("adec", encoding="utf-8")
        write(self.data / "vocab.json", dict(a=0,b=1,c=2,d=3,e=4))
        write(self.data / "alignment.json", dict(token_alignment=dict(bad_ids=[0,1,2],good_ids=[0,3,4,2],bad_text_sha256=builder.textsha("abc"),good_text_sha256=builder.textsha("adec"))))
        self.input = dict(schema=1,kind="sentence_target_input_v1",purpose="synthetic_test",source_split="synthetic_test",pair_id="synthetic_pair",expected_dim=8,
            x={**self.ref("x.npy"),"prompt_sha256":builder.textsha("abc")}, y={**self.ref("y.npy"),"prompt_sha256":builder.textsha("adec")},
            texts={"x":self.ref("x.txt"),"y":self.ref("y.txt")},alignment=self.ref("alignment.json"),tokenizer_vocab=self.ref("vocab.json"),
            blocks=[dict(x_token_span=[1,2],y_token_span=[1,3],x_text="b",y_text="de")],prefix_token_span=[0,1])
        write(self.data / "build.json", self.input)

    def ref(self, name):
        return dict(path=name, sha256=builder.sha(self.data / name))

    def prepare_fit(self):
        receipt = builder.read(self.data / "target/TARGET_RECEIPT.json")
        write(self.data / "context.json", dict(pair_id="synthetic_pair",split="train",encoder_id="synthetic_encoder_not_h3",
            prompt_x_sha256=builder.textsha("abc"),prompt_y_sha256=builder.textsha("adec"),
            x_raw_sha256=receipt["x_raw_sha256"],y_raw_sha256=receipt["y_raw_sha256"],source_precision=receipt["source_precision"]))
        row=dict(pair_id="synthetic_pair",split="train",work_id="synthetic_work",group_id="synthetic_family",
            prompt_x=self.ref("x.txt"),prompt_y=self.ref("y.txt"),context=self.ref("context.json"),target_receipt=self.ref("target/TARGET_RECEIPT.json"),arrays=self.ref("target/sentence_target.npz"),
            raw_sha256={k:receipt["arrays"][k]["raw_sha256"] for k in ("x","y","target")},
            masks={k:k for k in ("sentence_mask","outside_sentence_mask","prefix_mask")},region_weights={"all":1.})
        manifest=dict(schema=1,kind="explicit_train_target_manifest_v1",evidence_class="synthetic_test",pairs=[row])
        write(self.data / "fit.json",manifest)
        return manifest

    def cli(self, tool, *args):
        env={**os.environ,"CUDA_VISIBLE_DEVICES":"","OMP_NUM_THREADS":"2","MKL_NUM_THREADS":"2","PYTHONDONTWRITEBYTECODE":"1"}
        result=subprocess.run([sys.executable,"-B","-X","utf8",str(ROOT/"tools"/tool),*map(str,args)],cwd=self.root,env=env,capture_output=True,text=True,encoding="utf-8",timeout=120)
        self.assertEqual(result.returncode,0,result.stderr)
        return json.loads(result.stdout)

    def test_builder_fitter_cli_other_cwd_and_saved_public_apply(self):
        # Both manifest and its dependencies use portable relative paths from another cwd.
        self.cli("build_sentence_targets.py","--manifest","data/build.json","--output-dir","data/target")
        self.prepare_fit()
        dry=self.cli("fit_explicit_targets.py","--manifest","data/fit.json")
        self.assertEqual(dry["status"],"dry_run_validated_no_model_or_optimizer_created")
        self.assertEqual(dry["config"]["steps"],300)
        self.assertFalse((self.root/"fit").exists())
        result=self.cli("fit_explicit_targets.py","--manifest","data/fit.json","--manifest-sha",builder.sha(self.data/"fit.json"),"--execute","--output","fit","--steps","2","--hidden","8","--layers","1","--heads","2","--max-tokens","16")
        self.assertFalse(result["video_quality_evidence"])
        self.assertEqual(result["evidence_class"],"synthetic_test")
        model=load_bridge(result["weights"]["path"],device="cpu",dtype=torch.float32)
        x=torch.from_numpy(np.load(self.data/"x.npy").astype(np.float32)).unsqueeze(0)
        tags=torch.ones(x.shape[1],dtype=torch.int64)
        actual,report=apply_bridge_to_conditioning([[x,{"minimax_token_tags":tags,"sentinel":"kept"}]],model,alpha=1.)
        self.assertEqual(report.applied,1)
        self.assertTrue(torch.isfinite(actual[0][0]).all())
        self.assertEqual(actual[0][0].shape,x.shape)
        self.assertIs(actual[0][1]["minimax_token_tags"],tags)
        self.assertEqual(actual[0][1]["sentinel"],"kept")
        self.assertLessEqual(max(v["max_abs_error"] for v in result["reload_checks"]),1e-6)

    def test_source_hash_and_val_rejected_before_archive_loading(self):
        bad=copy.deepcopy(self.input);bad["x"]["sha256"]="0"*64
        write(self.data/"bad.json",bad)
        with self.assertRaisesRegex(ValueError,"SHA256 mismatch"):
            builder.build(self.data/"bad.json",self.data/"refused")
        self.assertFalse((self.data/"refused").exists())
        builder.build(self.data/"build.json",self.data/"target")
        manifest=self.prepare_fit()
        bad=copy.deepcopy(manifest);bad["pairs"][0]["split"]="val";write(self.data/"bad_fit.json",bad)
        with patch.object(fitter.np,"load",side_effect=AssertionError("archive opened")),self.assertRaisesRegex(ValueError,"train pairs"):
            fitter.load_manifest(self.data/"bad_fit.json",fitter.Config())
        bad=copy.deepcopy(manifest);bad["pairs"][0]["arrays"]["sha256"]="0"*64;write(self.data/"bad_fit.json",bad)
        with patch.object(fitter.np,"load",side_effect=AssertionError("archive opened")),self.assertRaisesRegex(ValueError,"changed bound file"):
            fitter.load_manifest(self.data/"bad_fit.json",fitter.Config())
        bad=copy.deepcopy(self.input);bad["source_split"]="val";write(self.data/"bad.json",bad)
        with self.assertRaisesRegex(ValueError,"refuses val/test"):
            builder.build(self.data/"bad.json",self.data/"refused_val")

    def test_receipt_relative_paths_and_native_dtype_preserved(self):
        receipt=builder.build(self.data/"build.json",self.data/"target")
        # Relocatable receipt bindings are interpreted relative to the receipt itself.
        receipt["output"]["path"]="sentence_target.npz"
        receipt["source_bindings"]={(os.path.relpath(p,self.data/"target") if Path(p).drive.lower()==self.data.drive.lower() else p):h for p,h in receipt["source_bindings"].items()}
        write(self.data/"target/TARGET_RECEIPT.json",receipt)
        self.prepare_fit()
        manifest,pairs=fitter.load_manifest(self.data/"fit.json",fitter.Config())
        self.assertEqual(pairs[0]["source_dtypes"]["x"],"float16")
        self.assertEqual(pairs[0]["source_dtypes"]["y"],"float32")
        fitter.verify_bindings(fitter.input_bindings(manifest,self.data/"fit.json",builder.sha(self.data/"fit.json")))
        with np.load(self.data/"target/sentence_target.npz") as z:
            mask=z["outside_sentence_mask"]
            self.assertEqual(z["target"][mask].tobytes(),z["monotone_target"][mask].tobytes())

if __name__=="__main__":unittest.main(verbosity=2)
