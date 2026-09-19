import json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from test_validate_deepseek_v4_conversion import make_fixture,converter,validator
import validate_deepseek_v4_recovery as final

class ReleasedStagingValidation(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        root=Path(self.tmp.name);self.source=make_fixture(root);self.model=root/'model'
        state=converter.convert(self.source,self.model,alignment=16,min_final_free=0,allow_fixture=True)
        groups,_=converter.build_plan(self.source,allow_fixture=True)
        headers={}
        for shard in self.source.glob('*.safetensors'):
            start,header=converter.read_safetensors_header(shard);headers[shard.name]={'start':start,'header':header,'size':shard.stat().st_size}
        (self.source/'recovery-headers.json').write_text(json.dumps({'revision':final.SOURCE_REVISION,'shards':headers}))
        proofs={}
        for group,records in groups.items():
            inventory=state['inventory'][group];segment=state['completed'][group]
            proofs[group]={'binding':{'revision':final.SOURCE_REVISION,'segment':segment,'inventory_sha256':final.digest_json(inventory)},
                           'validation':validator._validate_group(self.source,self.model,group,records,inventory,segment)}
        self.proofs=proofs;self.group=next(iter(proofs))
        self.proof_path=self.model/'recovery-validation.json';self.proof_path.write_text(json.dumps(proofs))
        (self.model/'historical-validation.json').write_text('{}')
        for shard in self.source.glob('*.safetensors'):shard.unlink()
    def run_check(self):return final.validate(self.source,self.model,allow_fixture=True)
    def test_rechecks_current_bytes_after_source_staging_was_released(self):
        result=self.run_check();self.assertEqual(result['status'],'passed');self.assertEqual(len(result['groups']),len(self.proofs))
    def test_missing_source_comparison_is_rejected(self):
        self.proofs[self.group]['validation']['source_bytes_compared']=0;self.proof_path.write_text(json.dumps(self.proofs))
        with self.assertRaisesRegex(ValueError,'comparison is incomplete'):self.run_check()
    def test_changed_inventory_binding_is_rejected(self):
        self.proofs[self.group]['binding']['inventory_sha256']='0'*64;self.proof_path.write_text(json.dumps(self.proofs))
        with self.assertRaisesRegex(ValueError,'does not bind'):self.run_check()
    def test_changed_output_bytes_are_rejected(self):
        path=self.model/self.group;data=bytearray(path.read_bytes());data[-1]^=1;path.write_bytes(data)
        with self.assertRaisesRegex(ValueError,'differs from source-verified bytes'):self.run_check()
    def test_changed_manifest_is_rejected(self):
        path=self.model/converter.MANIFEST_FILE;path.write_text(path.read_text()+' ')
        with self.assertRaisesRegex(ValueError,'manifest-bound'):self.run_check()
if __name__=='__main__':unittest.main()
