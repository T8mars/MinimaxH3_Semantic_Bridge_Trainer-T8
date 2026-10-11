"""Human-readable deployment settings; never infer video quality from text loss."""
from pathlib import Path
import math

from .apply import application_contract
from .trainer_data import sha256, write_json


def make_inference_guidance(model, training_config=None):
    extra = model.cfg.extra or {}
    contract = application_contract(model)
    cfg = training_config if training_config is not None else extra.get('training_application', {})
    if not isinstance(cfg, dict):
        raise ValueError('training_application must be a mapping')
    bounds = None
    if 'alpha_min' in cfg or 'alpha_max' in cfg:
        low, high = cfg.get('alpha_min'), cfg.get('alpha_max')
        if (type(low) not in (float, int) or type(high) not in (float, int)
                or not math.isfinite(low) or not math.isfinite(high) or not 0 < low <= high <= 1):
            raise ValueError('Invalid training alpha range')
        bounds = [float(low), float(high)]
    if contract:
        alpha = float(contract['alpha_min'])
        magnitude = contract['magnitude_match']
        status, basis = 'fixed_contract', 'weight_application_contract'
        trials = [alpha]
    else:
        alpha = .12 if bounds is None or bounds[0] <= .12 <= bounds[1] else sum(bounds)/2
        magnitude = cfg.get('magnitude_mode', 'per_token')
        if magnitude not in ('none', 'per_token'):
            raise ValueError('Unsupported training magnitude mode')
        status, basis = 'unvalidated_starting_point', 'training_formula' if cfg else 'legacy_fallback_no_training_recipe'
        trials = sorted(set([round(alpha, 8)] + [a for a in (.05, .12, .2) if bounds is None or bounds[0] <= a <= bounds[1]]))
    return {
        'schema': 1, 'status': status, 'basis': basis,
        'settings': {'alpha': alpha, 'magnitude_match': magnitude, 'token_scope': 'all_tokens',
                     'chunk_tokens': 0, 'auto_alpha': False, 'guard': False, 'allow_dim_mismatch': False},
        'training_alpha_range': bounds, 'trial_alphas': trials,
        'video_calibrated_by_this_export': False,
        'warning': 'Fixed contracts must be followed. Otherwise alpha is only a starting point, not a video-calibrated optimum. Training alpha ranges and text losses do not select the best video strength.',
    }


def write_inference_guidance(weight, model, training_config=None, output_dir=None):
    weight = Path(weight)
    folder = Path(output_dir) if output_dir is not None else weight.parent
    folder.mkdir(parents=True, exist_ok=True)
    guidance = make_inference_guidance(model, training_config)
    record = {**guidance, 'model_file': weight.name, 'model_sha256': sha256(weight)}
    write_json(folder/'inference_settings.json', record)
    s = guidance['settings']
    fixed = guidance['status'] == 'fixed_contract'
    alpha_note = (f"**固定 alpha = {s['alpha']:g}，必须按权重契约使用。**" if fixed
                  else f"**建议从 alpha = {s['alpha']:g} 开始试用；这不是已验证的最佳值，也不是必须固定的值。**")
    bounds = guidance['training_alpha_range']
    training_note = (f"训练时 alpha 在 {bounds[0]:g}–{bounds[1]:g} 之间采样。该范围是训练配方，不等于最佳视频强度。" if bounds else "该权重未附可识别的训练 alpha 范围；不要据此猜测训练方式。")
    if fixed:
        next_step = '此模型带固定应用契约；不要改成其他模型的强度。固定为1的旧Sentence75与普通范围混合配方不同。参数固定不代表任意提示词都有效。'
    else:
        next_step = ('不需要在0.12和1之间猜一个。先保持下表整套公式，按同提示词、同seed做关桥/开桥小样；可试 alpha = '
                     + ' / '.join(f'{a:g}' for a in guidance['trial_alphas'])
                     + '。根据动作顺序、空间、遮挡和归属决定是否保留；不要只凭特效亮度或文本loss选强度。未做视频对照前，不标成“最佳alpha”。')
    text = f'''# 本模型怎么用 · By T8star

模型：`{weight.name}`

{alpha_note}

| T8节点选项 | 使用值 |
|---|---|
| alpha | {s['alpha']:g}（{'固定契约' if fixed else '未校准的试用起点'}） |
| magnitude_match | `{s['magnitude_match']}` |
| token_scope | `all_tokens` |
| chunk_tokens | `0` |
| auto_alpha / guard | 关闭 |

把权重放到自己ComfyUI的`models/semantic_bridge/t8_compat/`，在T8节点中选择。使用说明不会自动修改另一项目的节点设置，请按表填写。

{training_note}

{next_step}

分发时把本说明、`inference_settings.json`与权重一起提供。JSON内SHA-256用于确认说明对应同一份模型；更名不影响内容哈希。不要沿用另一模型的说明。

SHA-256：`{record['model_sha256']}`。

这是Semantic Bridge文本条件适配器，不是视频LoRA。自动导出说明没有完成视频校准或效果验收。
'''
    target = folder/'模型使用说明.md'
    temporary = target.with_suffix('.md.tmp')
    temporary.write_text(text, encoding='utf-8')
    temporary.replace(target)
    return record
