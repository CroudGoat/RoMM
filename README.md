# RoMM (Routing of Model Merging)

パラメータの幾何学的特徴（コサイン類似度・ノルム）に基づいて、レイヤーごとに最適なマージ手法・マージパラメータを自動選定する、モデルマージ向けルーティングツールです。

## 背景

[Mergekit](https://github.com/arcee-ai/mergekit) が提供する SLERP や DARE-TIES などのマージ手法は train-free で高速・軽量なため広く使われていますが、どのレイヤーにどの手法・パラメータを適用するかはユーザーの経験則に依存しがちです。かといって EvoLLM のような自動探索はベンチマーク実行や複数モデル推論を要し、一般ユーザーには重すぎます。

RoMM は、モデルA・モデルBそれぞれのタスクベクトル（ベースモデルとの差分）をレイヤーごとに解析し、**統計的に有意な幾何構造の違い**からマージ手法とパラメータを自動決定します。

## 仕組み

1. **特徴抽出**: レイヤーごとにモデルA・BのタスクベクトルΔ_A, Δ_Bを求め、コサイン類似度とノルムを算出
2. **実効次元の推定**: べき乗法（Power Iteration）でStable Rank（実効次元）を推定し、物理次元数によるZスコアのインフレを補正
3. **統計的ルーティング**: 実効次元で正規化したZスコアに基づき、レイヤーを3領域に分類
   - **High-Similarity**（Z ≥ +2.5σ）→ **SLERP**
   - **Orthogonal / Moderate**（-2.0σ ≤ Z < +2.5σ）→ **弱スパース DARE-TIES**
   - **Conflicting**（Z < -2.0σ）→ **強スパース DARE-TIES**
4. **パラメータ決定**: ノルム比からSLERPの補間係数 `t` およびDARE-TIESの `weight`（ガードレール付き）、衝突度に応じた `density` を算出
5. **出力**: CSV / JSON の解析結果、および Mergekit がそのまま利用できるYAML設定ファイルを生成

理論の詳細な数式・導出は [`THEORY.md`](./THEORY.md) を参照してください。

## セットアップ

```bash
pip install torch safetensors huggingface_hub pyyaml
```

## 使い方

```bash
python scripts/router.py \
  --base <ベースモデルのrepo_idまたはローカルパス> \
  --model-a <モデルAのrepo_idまたはローカルパス> \
  --model-b <モデルBのrepo_idまたはローカルパス>
```

実行すると `results/` 以下に以下のファイルが出力されます（`--output-dir` で変更可）。

- `<base>__<A>_x_<B>_routing.csv` : レイヤーごとの解析結果
- `<base>__<A>_x_<B>_routing.json` : 同上（設定情報付き）
- `<base>__<A>_x_<B>_mergekit.yaml` : Mergekitにそのまま渡せるマージ設定

### 主なオプション

| オプション | 説明 | デフォルト |
|---|---|---|
| `--output-dir` | 出力先ディレクトリ | `results` |
| `--no-yaml` | YAML生成をスキップ | 生成する |
| `--offline` | ローカルキャッシュのみ使用 | オフ |
| `--chunk-elements` | テンソルストリーミングのチャンクサイズ | `2000000` |
| `--use-stable-rank` / `--no-use-stable-rank` | Stable Rankによる実効次元補正の有無 | 有効 |
| `--power-iters` | スペクトルノルム推定のべき乗法反復回数 | `5` |
| `--z-slerp` | High-Similarity判定のZ閾値（σ） | `2.5` |
| `--z-conflict` | Conflicting判定のZ閾値（σ） | `2.0` |
| `--base-density` | DARE-TIESの基準density | `0.20` |
| `--alpha-a` / `--alpha-b` | モデルA/Bのユーザー優先度 | `1.0` / `1.0` |
| `--layer-pattern` | レイヤー番号抽出用の正規表現 | `(?:^|\.)layers\.(\d+)(?:\.|$)` |

`--base` / `--model-a` / `--model-b` には Hugging Face の repo_id、もしくはローカルディレクトリのパスを指定できます（`safetensors` 形式のみ対応）。

生成されたYAMLは、そのまま Mergekit に渡してマージを実行できます。

```bash
mergekit-yaml results/<base>__<A>_x_<B>_mergekit.yaml ./output-model
```

## ディレクトリ構成

```
RoMM/
├── THEORY.md          # 理論・数式の詳細
├── ISSUE-STATUS.md     # 既知の課題・改善案（理論面／実装面）
└── scripts/
    └── router.py       # ルーティング・パラメータ算出の実行スクリプト
```

## 既知の課題

現時点でいくつかの理論的・実装的な課題が判明しています。詳細と改善案は [`ISSUE-STATUS.md`](./ISSUE-STATUS.md) を参照してください。主なもの:

- 学習変化量が極小のレイヤーへのノルム逆数補正によるノイズ増幅リスク
- レイヤー間でのマージ手法の不連続な切り替わりによるResidual Streamへの影響
- Attention/MLPを一括集約することによる幾何情報の希釈
- `embed_tokens` / `lm_head` など非Transformer層の扱い
- ゼロ除算・極小値に対するロバストネス
- 可視化・診断機能の不足
- Mergekit依存によるテンソル単位ハイブリッドマージの不可能性（将来的に `romm-engine` として自前実装を計画）

## ライセンス

未定
