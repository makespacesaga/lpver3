"""
パーソナルジムLP 第3版（v3.html）用の画像生成（OpenAI gpt-image-2）

使い方:
  python generate_hero.py                 # 下の JOBS を全部生成
  python generate_hero.py hero_a hero_b   # 名前を指定して生成
  python generate_hero.py --out <dir>     # 保存先（既定: images/gen/）

鍵は LP/.env の OpenAI_apikey を読む（値は表示しない）。
"""
import base64
import concurrent.futures as cf
import json
import os
import sys
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))

# ── 共通の「AI感を消す」指示 ───────────────────────────────────────────
REAL = (
    "Candid documentary-style editorial photograph, looks like a real unretouched photo. "
    "Natural skin texture with pores and fine lines, no airbrushing, no glossy skin, "
    "slightly imperfect asymmetry, natural hair with a few stray strands. "
    "Shot on a 35mm lens at f/2.8, soft natural window light, gentle film grain, "
    "muted warm color grading, shallow depth of field. "
    "No text, no logos, no watermark, no posed stock-photo look."
)

ROOM = (
    "Setting: a small private one-on-one personal training studio in Japan with raw grey concrete walls, "
    "warm natural wood accents, dark stone-look floor, black-and-yellow TRX suspension straps hanging from a rig, "
    "a large window letting in soft daylight."
)

JOBS = {
    # ヒーロー候補A：TRX中、トレーナーの手が軽くサポート（顔は出さない）
    "hero_a": dict(size="1024x1024", quality="high", prompt=(
        f"{REAL} {ROOM} "
        "Subject: a Japanese woman in her mid-40s with an average, slightly soft build (not a fitness model), "
        "hair loosely tied back, wearing a plain charcoal T-shirt and dark leggings. "
        "She is doing a gentle TRX row, leaning back holding the straps, a genuine relaxed smile, slightly out of breath, "
        "clearly enjoying it. At the edge of the frame, only the forearm and hand of a trainer in a white polo shirt "
        "lightly supports her lower back; the trainer's face is not visible. "
        "Composition: the woman centered, waist-up to full body, eye-level, space around her."
    )),
    # ヒーロー候補B：セッション後、タオルを肩にかけてほっとした笑顔
    "hero_b": dict(size="1024x1024", quality="high", prompt=(
        f"{REAL} {ROOM} "
        "Subject: a Japanese woman in her early 40s with an average build, natural minimal makeup, "
        "a little flushed after exercise, a small white towel over one shoulder, holding a bottle of water. "
        "She is laughing naturally, relieved and happy, looking slightly off-camera as if talking to her trainer. "
        "Composition: chest-up portrait, centered, soft blurred TRX straps and concrete wall behind her."
    )),
    # ヒーロー候補B2：Bと同じ場面、手元を枠外にして腰から上の構図。前景にトレーナーの肩（ぼかし）
    "hero_b2": dict(size="1024x1024", quality="high", prompt=(
        f"{REAL} {ROOM} "
        "Subject: a Japanese woman in her mid-40s with an average, slightly soft build (not slim, not a fitness model), "
        "natural minimal makeup, a little flushed after exercise, a small white towel over one shoulder. "
        "She is laughing naturally, relieved and happy, looking slightly off-camera to the left as if talking to her trainer. "
        "Her hands are below the frame. In the near foreground at the left edge, the out-of-focus shoulder of a trainer "
        "in a white polo shirt, back to camera, face not visible. "
        "Composition: waist-up, the woman slightly right of center, soft blurred TRX straps and concrete wall behind her."
    )),
    # ヒーロー候補B3：より平均的な体型・丸みのある顔立ち。窓辺のベンチで水を飲みながら笑う
    "hero_b3": dict(size="1024x1024", quality="high", prompt=(
        f"{REAL} {ROOM} "
        "Subject: a Japanese woman around 48 with a noticeably average, softly rounded build and a round friendly face "
        "(an ordinary office worker, not athletic), hair in a loose low ponytail with stray strands, "
        "wearing a loose heather-grey T-shirt. She sits on a wooden bench by the window after her session, "
        "towel on her lap, laughing with her eyes closed in genuine relief. "
        "Composition: chest-up, centered, warm window light from the left, blurred TRX rig behind."
    )),
    # ヒーロー候補C：マットでストレッチ、トレーナーが姿勢を軽く整える（顔は出さない）
    "hero_c": dict(size="1024x1024", quality="high", prompt=(
        f"{REAL} {ROOM} "
        "Subject: a Japanese woman around 45 with an average build, sitting on a dark exercise mat doing a seated side stretch, "
        "calm content expression with a soft smile. Only the hands of a trainer in a white polo gently guide her shoulder "
        "and upper back; the trainer's face is outside the frame. "
        "Composition: the woman centered, full body visible, eye-level, quiet and reassuring mood."
    )),
    # バナー用：3食きちんと食べる（和定食）
    "meal": dict(size="1536x1024", quality="high", prompt=(
        f"{REAL} "
        "Overhead-angled photograph of a balanced Japanese home-style meal (ichiju-sansai) on a wooden table: "
        "a bowl of white rice, miso soup, grilled salmon, a small dish of simmered vegetables, "
        "a side of leafy greens with sesame, and a cup of green tea. Natural daylight from a window, "
        "simple ceramic dishes, a linen cloth. Appetizing, realistic, home-cooked, not restaurant-styled. "
        "No text, no hands, no people."
    )),
}


def load_key():
    env_path = os.path.join(HERE, "..", ".env")
    if not os.path.exists(env_path):
        sys.exit(f"[ERROR] .env が見つかりません: {env_path}")
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("OpenAI_apikey="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    sys.exit("[ERROR] OpenAI_apikey が .env に見つかりません")


def generate(api_key, name, job, out_dir):
    body = json.dumps({
        "model": "gpt-image-2",
        "prompt": job["prompt"],
        "n": 1,
        "size": job["size"],
        "quality": job["quality"],
    }).encode("utf-8")
    req = urllib.request.Request(
        "https://api.openai.com/v1/images/generations",
        data=body,
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json; charset=utf-8"},
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            result = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return name, f"HTTP {e.code}: {e.read().decode()[:300]}"
    except Exception as e:  # noqa: BLE001
        return name, f"{type(e).__name__}: {e}"
    b64 = result["data"][0].get("b64_json")
    if not b64:
        return name, f"b64_json がありません: {str(result)[:200]}"
    path = os.path.join(out_dir, f"{name}.png")
    with open(path, "wb") as f:
        f.write(base64.b64decode(b64))
    return name, f"OK {os.path.getsize(path)//1024} KB -> {path}"


def main():
    args = sys.argv[1:]
    out_dir = os.path.join(HERE, "images", "gen")
    if "--out" in args:
        i = args.index("--out")
        out_dir = args[i + 1]
        del args[i:i + 2]
    names = args or list(JOBS)
    unknown = [n for n in names if n not in JOBS]
    if unknown:
        sys.exit(f"[ERROR] 不明なジョブ名: {unknown}（使える名前: {list(JOBS)}）")
    os.makedirs(out_dir, exist_ok=True)
    key = load_key()
    print(f"生成: {names} → {out_dir}")
    with cf.ThreadPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(generate, key, n, JOBS[n], out_dir) for n in names]
        for fut in cf.as_completed(futs):
            name, msg = fut.result()
            print(f"  [{name}] {msg}")


if __name__ == "__main__":
    main()
