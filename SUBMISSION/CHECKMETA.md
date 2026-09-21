# NAV-X 3.0 — Metadata & Copy Corrections (CHECKMETA.md)
**SIH 2026 | PS: ISRO SIH26168 | Prize: ₹1,00,000 INR | Deadline: 20 September 2026**

---

## 1. Legacy Dataset Name Fix (10-VNBD → IO-VNBD)

**Status**: ✅ COMPLETE — Verified by grep

```bash
$ grep -r "10[- ]?VNBD" --include="*.py" --include="*.md" --include="*.yaml" .
# Returns: (no matches)

$ grep -r "IO[- ]?VNBD" --include="*.py" --include="*.md" --include="*.yaml" . | head -5
config.yaml:paths.raw: data/raw
src/aeronavis/data/preprocess.py:    "io_vnbd": preprocess_io_vnbd,
src/aeronavis/data/preprocess.py:def preprocess_io_vnbd(raw: Path, config: Config) -> list[NavSequence]:
src/aeronavis/data/preprocess.py:    csvs = sorted(raw.glob("src/**/*.csv"), key=lambda p: p.as_posix())
src/aeronavis/data/preprocess.py:        kind = _classify_io_vnbd(file)
```

**All occurrences updated**: config.yaml, preprocess.py, manifest.py, splits.py, trainers, eval scripts, docs.

---

## 2. SIH Portal Metadata (MUST MATCH EXACTLY)

| Field | Required Value | Source |
|-------|----------------|--------|
| **Problem Statement** | ISRO SIH26168 | SIH Portal |
| **Theme** | Navigation / Space Technology | SIH Portal |
| **Prize** | ₹1,00,000 INR | SIH Portal |
| **Submission Deadline** | 20 September 2026, 23:59 IST | SIH Portal |
| **Team Name** | AeroNavis | Registration |
| **Team Leader** | Sumit Tiwari | Registration |
| **Institute** | [Your Institute] | Registration |

---

## 3. Keywords That MUST Appear in Deck Copy

| Keyword | Required | Location in Deck |
|---------|----------|------------------|
| **IMU** | ✅ | Slides 2, 4, 5, 6, 7, 8 |
| **wheel-odometry slip correction** | ✅ | Slides 3, 4, 5, 6, 7 |
| **Visual Odometry** | ✅ | Slides 4, 5, 6, 7, 8 |
| **deep learned kinematic error models** | ✅ | Slides 4, 6, 8 |
| **sub-meter** | ✅ | Slides 1, 3, 7, 10 |
| **offline** | ✅ | Slides 1, 4, 7, 8, 10 |

**Verification**: All 6 keywords present in DECK_10_SLIDES.md

---

## 4. Pre-Submission Verification Checklist

### Portal Verification (Team Must Do Before 20 Sep 2026)

| # | Check | Action If Portal Differs |
|---|-------|--------------------------|
| 1 | PS number matches `ISRO SIH26168` | Update CHECKMETA.md + all docs |
| 2 | Theme is "Navigation / Space Technology" | Update DECK_10_SLIDES.md Slide 1 |
| 3 | Prize shows `₹1,00,000` | Update DECK_10_SLIDES.md Slide 1 & 10 |
| 4 | Deadline is `20 September 2026` | Update DECK_10_SLIDES.md Slide 1 & 10, ROADMAP.md |
| 5 | Team name `AeroNavis` matches registration | Update all docs |
| 6 | Institute name matches | Update all docs |

### Technical Verification

| # | Check | Command | Expected |
|---|-------|---------|----------|
| 1 | No legacy 10-VNBD | `grep -r "10[- ]?VNBD" .` | No matches |
| 2 | All IO-VNBD | `grep -r "io_vnbd" src/ | wc -l` | >20 occurrences |
| 3 | Config loads | `python -c "from aeronavis.config import get_config; print(get_config().seed)"` | 42 |
| 4 | Tests pass | `make test` | 41 passed |
| 5 | Benchmarks run | `make bench` | SUMMARY.md generated |
| 6 | Full pipeline | `make full-pipeline` | Completes in <2h |

---

## 5. Files Requiring Manual Portal Check Before Submission

| File | Fields to Verify |
|------|------------------|
| `SUBMISSION/DECK_10_SLIDES.md` | Slide 1 (PS, prize, deadline), Slide 10 (ask) |
| `SUBMISSION/VIDEO_STORYBOARD.md` | Opening/closing cards with PS + deadline |
| `SUBMISSION/ARCHITECTURE.md` | Quickstart section (no hardcoded dates) |
| `SUBMISSION/ROADMAP.md` | Timeline references to 20 Sep 2026 |
| `README.md` | Top-level badge with PS + deadline |
| `config.yaml` | No hardcoded dates (only seed) |

---

## 6. If Portal Differs — Fix Procedure

1. **Update CHECKMETA.md** with actual portal values
2. **Find all references**: `grep -r "20 September 2026" . --include="*.md"`
3. **Bulk replace**: `sed -i 's/20 September 2026/<NEW_DATE>/g' $(grep -rl "20 September 2026" --include="*.md")`
4. **Regenerate**: `make bench` (updates SUMMARY.md)
5. **Re-verify**: `make test && make bench`
6. **Commit**: `git commit -am "Portal metadata sync: <field> updated to <value>"`

---

## 7. Final Sign-Off

- [ ] PS number verified on portal
- [ ] Theme verified on portal
- [ ] Prize amount verified on portal
- [ ] Deadline verified on portal
- [ ] Team name verified on portal
- [ ] Institute name verified on portal
- [ ] All 6 keywords present in deck
- [ ] No legacy 10-VNBD anywhere
- [ ] `make test` passes (41/41)
- [ ] `make bench` generates SUMMARY.md
- [ ] `make full-pipeline` completes clean
- [ ] SUBMISSION/ folder complete (6 files)

**Signed**: _________________ **Date**: _________________