from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import content_history as ch  # noqa: E402
from gemini_client import ContentPlan, Slide, facts_text, normalize_plan, select_plan  # noqa: E402
from text_case import sentence_case, title_case  # noqa: E402


def make_plan(topic_id="kep-vs-eposta", pillar="kep_bilgi", headline="KEP mi, e-posta mı?", caption=None):
    return ContentPlan(
        topic_id=topic_id,
        pillar=pillar,
        format="single",
        slides=[Slide(headline=headline)],
        caption=caption or "KEP ile resmi yazışmalarınız yasal güvence altında. Bize DM atın.",
    )


def entry(plan):
    return ch.build_entry(plan)


FACTS = facts_text()


def test_normalize_keeps_turkish_letters():
    assert ch.normalize_text("KEP mi, E-POSTA mı?") == "kep mi e posta mı"
    assert ch.normalize_text("İmza IŞIK Şirket") == "imza ışık şirket"


def test_same_topic_is_rejected():
    history = [entry(make_plan())]
    verdict = ch.check_plan(make_plan(headline="Tamamen farklı başlık", pillar="sss"), history, FACTS, cooldown=0)
    assert verdict.exact_repeat
    assert not verdict.clean


def test_same_headline_different_topic_is_rejected():
    history = [entry(make_plan())]
    plan = make_plan(topic_id="kep-nedir", pillar="sss", headline="kep mi  E-POSTA mı")
    assert ch.check_plan(plan, history, FACTS, cooldown=0).exact_repeat


def test_consecutive_same_pillar_is_rejected():
    history = [entry(make_plan())]
    plan = make_plan(topic_id="kep-nedir", headline="KEP nedir?")
    verdict = ch.check_plan(plan, history, FACTS, cooldown=2)
    assert verdict.pillar_repeat
    assert verdict.only_pillar_violation


def test_pillar_outside_cooldown_window_is_allowed():
    history = [
        entry(make_plan()),
        entry(make_plan(topic_id="eimza-nedir", pillar="eimza_bilgi", headline="E-imza nedir?")),
        entry(make_plan(topic_id="sss-dkr-nedir", pillar="sss", headline="DKR kimdir?")),
    ]
    plan = make_plan(topic_id="kep-nedir", headline="KEP nedir?")
    assert ch.check_plan(plan, history, FACTS, cooldown=2).clean


@pytest.mark.parametrize(
    "caption",
    [
        "KEP başvurusu sadece 299 TL!",
        "E-imzada %20 indirim fırsatı.",
        "Başvurunuz 2 saatte tamamlanır.",
        "KEP adresiniz 20 dakikada hazır.",
        "Şimdi 3 adımda başvurun.",
        "Bu ay ücretsiz kurulum.",
        "Fiyat bilgisi için arayın: 1500₺",
    ],
)
def test_fabricated_claims_are_rejected(caption):
    plan = make_plan(caption=caption)
    verdict = ch.check_plan(plan, [], FACTS, cooldown=0)
    assert verdict.fabricated, caption
    assert not verdict.clean


@pytest.mark.parametrize(
    "caption",
    [
        "KEP delilleri 20 yıl boyunca saklanır.",
        "Başvurudan sonra 7/24 teknik destek veriyoruz.",
        "4 adımda kolay başvuru. #kep2026",
    ],
)
def test_facts_backed_numbers_are_allowed(caption):
    assert ch.find_fabricated_claims(caption, FACTS) == []


def test_select_plan_falls_back_to_pillar_only_violation():
    history = [entry(make_plan())]
    candidates = iter([
        make_plan(topic_id="kep-nedir", headline="KEP nedir?"),
        make_plan(topic_id="kep-kesin-delil", headline="Mahkemede kesin delil"),
    ])
    plan = select_plan("post", lambda note: next(candidates), history, retries=2)
    assert plan.topic_id == "kep-nedir"


def test_select_plan_never_returns_exact_repeat_or_fabrication():
    history = [entry(make_plan())]
    candidates = iter([
        make_plan(),
        make_plan(topic_id="eimza-nedir", pillar="eimza_bilgi", headline="E-imza nedir?", caption="Sadece 99 TL"),
    ])
    with pytest.raises(RuntimeError):
        select_plan("post", lambda note: next(candidates), history, retries=2)


def test_select_plan_passes_rejection_note():
    history = [entry(make_plan())]
    notes = []
    plans = iter([make_plan(), make_plan(topic_id="eimza-nedir", pillar="eimza_bilgi", headline="E-imza nedir?")])

    def produce(note):
        notes.append(note)
        return next(plans)

    assert select_plan("post", produce, history, retries=3).topic_id == "eimza-nedir"
    assert notes[0] == "" and "kep-vs-eposta" in notes[1]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("KEP Mİ, E-POSTA MI?", "KEP mi, e-posta mı?"),
        ("Islak İmzayla Aynı Geçerlilik", "Islak imzayla aynı geçerlilik"),
        ("e-imzanız E-DEVLET ile uyumlu", "E-imzanız e-Devlet ile uyumlu"),
        ("acıbadem'deyiz. hemen ARAYIN!", "Acıbadem'deyiz. Hemen arayın!"),
        ("türkkep YETKİLİ başvuru merkezi", "TÜRKKEP yetkili başvuru merkezi"),
        ("işletmeniz için KEP", "İşletmeniz için KEP"),
        ("DKR TÜRKKEP BAŞVURU MERKEZİ kimdir?", "DKR TÜRKKEP Başvuru Merkezi kimdir?"),
    ],
)
def test_sentence_case(raw, expected):
    assert sentence_case(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("KEP ile standart e-posta arasındaki farklar", "KEP ile Standart E-Posta Arasındaki Farklar"),
        ("geleceğe hazır MISINIZ?", "Geleceğe Hazır mısınız?"),
        ("e-imza ile ıslak imza aynı geçerlilikte", "E-İmza ile Islak İmza Aynı Geçerlilikte"),
        ("acıbadem'deyiz", "Acıbadem'deyiz"),
        ("KEP'in avantajları", "KEP'in Avantajları"),
        ("e-devlet ile uyumlu e-imza", "E-Devlet ile Uyumlu E-İmza"),
        ("ve sonra KEP", "Ve Sonra KEP"),
        ("dkr türkkep başvuru merkezi kimdir?", "DKR TÜRKKEP Başvuru Merkezi Kimdir?"),
    ],
)
def test_title_case(raw, expected):
    assert title_case(raw) == expected


def test_image_model_discovery_prefers_stable_flash_image():
    from image_gen import discover_image_model

    available = ["gemini-3.5-flash", "gemini-x-flash-image-preview", "gemini-x-flash-image", "imagen-x-generate"]
    assert discover_image_model(available) == "gemini-x-flash-image"
    assert discover_image_model(["imagen-x-generate", "gemini-3.5-flash"]) == "imagen-x-generate"
    assert discover_image_model(["gemini-3.5-flash", "gemini-embedding-001"]) is None


def test_history_is_trimmed(tmp_path, monkeypatch):
    monkeypatch.setenv("CONTENT_HISTORY_SIZE", "3")
    path = tmp_path / "post_history.json"
    for i in range(5):
        ch.record_published("post", make_plan(topic_id=f"t{i}", headline=f"Başlık {i}"), path=path)
    entries = ch.load_history("post", path)
    assert [e["topic_id"] for e in entries] == ["t2", "t3", "t4"]


def test_normalize_plan_forces_story_and_carousel_rules():
    data = {
        "topic_id": "kep-nedir",
        "pillar": "yanlis",
        "format": "carousel",
        "slides": [{"headline": f"Slayt {i}"} for i in range(12)],
        "caption": "Metin",
        "hashtags": ["KEP", "#Eİmza"],
    }
    story = normalize_plan(data, "story")
    assert story.format == "story" and len(story.slides) == 1 and story.pillar == "kep_bilgi"

    carousel = normalize_plan(data, "post")
    assert carousel.format == "carousel" and len(carousel.slides) == 10
    assert carousel.slides[-1].headline == "Başvuru İçin Bize Ulaşın"
    assert carousel.hashtags == ["#kep", "#eimza"]

    with pytest.raises(ValueError):
        normalize_plan({**data, "slides": [{"headline": "Tek"}]}, "post", "carousel")
    with pytest.raises(ValueError):
        normalize_plan({**data, "topic_id": "uydurma-konu"}, "post")
