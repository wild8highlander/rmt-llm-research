# Запуск эксперимента масштабирования на Android: Termux + proot-distro Ubuntu

> Пошаговая инструкция, как запустить A/B-эксперимент «формулы vs галлюцинации»
> (`research/tinygpt_formula/scaling/`) прямо на телефоне: Termux → proot-distro →
> Ubuntu → Python 3 + NumPy → обучение TinyGPT v3 (S 447K, опционально M ~1.1M).
> Всё бесплатно, без root, без внешнего сервера.

## 0. Почему через proot-distro Ubuntu, а не «голый» Termux

| | Termux (native) | Termux + proot Ubuntu |
|---|---|---|
| Python/NumPy | `pkg install python-numpy` (сборка под bionic) | `apt install python3-numpy` (официальные deb-пакеты Debian/Ubuntu arm64) |
| Совместимость скриптов | иногда отличаются пути/символы | **полный glibc, как на ПК/CI** |
| git, make, компиляторы | да, но пакеты Termux | да, из стандартного репозитория |
| Скорость | ~100% | ~90-95% (proot-оверход syscall) |

Эксперимент требует только Python 3.10+ и NumPy, поэтому оба пути рабочие;
proot-distro даёт предсказуемую desktop-среду и удобен для пуша на GitHub.

## 1. Установка Termux

1. Ставьте Termux **из F-Droid** (https://f-droid.org) или с GitHub Releases
   (github.com/termux/termux-app/releases). **Версия из Google Play устарела и не поддерживается.**
2. Первый запуск: разрешите уведомления (нужны для wakelock-сессии).

## 2. Доступ к общей памяти (для распаковки ZIP из Downloads)

```bash
termux-setup-storage          # разрешите доступ к хранилищу
```
После этого `~/storage/downloads` = папка Downloads телефона.

## 3. Установка Ubuntu через proot-distro

```bash
pkg update -y && pkg upgrade -y
pkg install -y proot-distro    # менеджер гостевых дистрибутивов
proot-distro install ubuntu    # ~25 MB, без root
proot-distro login ubuntu      # вход в Ubuntu (приглашение сменится на root@localhost)
```

Выход из Ubuntu обратно в Termux: команда `exit`.

## 4. Внутри Ubuntu: Python + NumPy + git

```bash
apt update -y
apt install -y python3 python3-numpy python3-pip git coreutils
python3 -c "import numpy; print('numpy', numpy.__version__)"
```

`python3-numpy` из apt — готовый пакет, компиляция не требуется.

## 5. Получить репозиторий — два способа

### Способ A: git clone (если scaling/ уже запушен на GitHub)
```bash
git clone --depth 1 https://github.com/wild8highlander/rmt-llm-research.git
cd rmt-llm-research/research/tinygpt_formula
```

### Способ B: распаковать ZIP-пакет rmt-llm-push.zip из Downloads
```bash
# в Termux (до входа в Ubuntu): ZIP из Downloads виден через ~/storage/downloads
pkg install -y unzip
cp ~/storage/downloads/rmt-llm-push.zip ~/
cd ~ && unzip -q rmt-llm-push.zip
# внутри: rmt-llm-push/rmt-llm-research + push_to_github.sh + INSTRUCTIONS_TERMUX.md

# вход в Ubuntu и переход к папке (Termux home виден в proot как /host-rootfs/home/...):
proot-distro login ubuntu --shared-tmp
cd /host-rootfs/data/data/com.termux/files/home/rmt-llm-push/rmt-llm-research/research/tinygpt_formula
```
Если путь `/host-rootfs/...` недоступен — просто скопируйте папку внутрь Ubuntu-домash:
```bash
# в Termux:
proot-distro login ubuntu -- mkdir -p /root/work
cp -r ~/rmt-llm-push /sdcard/Download/ 2>/dev/null || true
# затем в Ubuntu: cp -r /sdcard/Download/rmt-llm-push /root/work/ (нужен termux-setup-storage + bind)
```
Проверенный минимальный вариант: `pkg install -y nano` не нужен — просто держите
ZIP в `~/` и распаковывайте ВНУТРИ Ubuntu после копирования через `/sdcard`:
```bash
# Termux: cp ~/rmt-llm-push.zip /sdcard/Download/
# Ubuntu: apt install -y unzip && cp /sdcard/Download/rmt-llm-push.zip . && unzip -q rmt-llm-push.zip
```

## 6. Запуск эксперимента (S — базовая точка, ~20-40 мин на руку на телефоне)

```bash
cd research/tinygpt_formula   # (или scaling/ целиком — см. путь выше)

# 1) контрольный корпус (секунды)
python3 scaling/build_control_corpus.py

# 2) быстрая проверка пайплайна (2 эпохи, ~2-3 мин)
SCALING_SMOKE=1 python3 scaling/scaling_experiment.py
rm -rf scaling/runs           # smoke-результаты в сводку не идут

# 3) настоящий прогон S: обе руки (прерывание безопасно — resume)
SCALING_CELLS=S_formula,S_control SCALING_MAX_MINUTES=38 python3 scaling/scaling_experiment.py
# не уложились в бюджет — просто запустите ту же команду ещё раз (продолжит)

# результаты:
cat scaling/SCALING_RESULTS.md
```

Опционально ночью — размер M (~1.1M): оставьте телефон на зарядке:
```bash
# в Termux (до входа в Ubuntu) запретите засыпание:
termux-wake-lock
# затем в Ubuntu:
SCALING_CELLS=M_formula,M_control SCALING_MAX_MINUTES=480 python3 scaling/scaling_experiment.py
# после завершения (в Termux): termux-wake-unlock
```

## 7. Запушить результаты на GitHub из proot-Ubuntu

```bash
# внутри Ubuntu:
apt install -y git
cd rmt-llm-research
git config user.name  "wild8highlander"
git config user.email "wild8highlander@users.noreply.github.com"
git add research/tinygpt_formula/scaling/SCALING_RESULTS.md research/tinygpt_formula/scaling/scaling_report.json
git commit -m "feat(scaling): S-point A/B results from Termux/Ubuntu run"
git remote -v   # origin должен указывать на ваш репозиторий
# push с Personal Access Token (создайте на github.com → Settings → Developer settings → PAT, права repo):
git push https://<ВАШ_PAT>@github.com/wild8highlander/rmt-llm-research.git main
```
Токен в командной строке не сохраняется в конфиге git. Либо используйте готовый
`push_to_github.sh` из ZIP-пакета (он валидирует токен и ищет репозиторий сам).

## 8. Что ожидать по времени (ориентиры)

| Этап | Флагман 2020-х (8 ядер) | Средний телефон (4-6 ядер) |
|---|---|---|
| SMOKE (2 эпохи S) | ~2 мин | ~4 мин |
| S_formula (40 эпох) | ~15-20 мин | ~30-40 мин |
| S_control (40 эпох) | ~15-20 мин | ~30-40 мин |
| M (60 эпох, обе руки) | ~3-5 ч | ночь на зарядке |
| L / XL | только Colab/Kaggle | только Colab/Kaggle |

## 9. Troubleshooting

- **`proot-distro: command not found`** — выполните `pkg update` и `pkg install proot-distro` ещё раз.
- **`numpy` не ставится в нативном Termux** — используйте proot-Ubuntu (п. 3-4), там ставится из apt.
- **Телефон греется / батарея тает** — норма для 40 эпох; ставьте на зарядку, `termux-wake-lock`, снимите чехол.
- **Android убивает сессию** — держите Termux в фоне с wakelock, не свайпайте из recents; большая батарея + отключённая оптимизация батареи для Termux в настройках Android.
- **`/sdcard` пуст в Ubuntu** — выполните `termux-setup-storage` в Termux и заходите через `proot-distro login ubuntu --bind /sdcard` (или скопируйте файлы через `~/storage`).
- **prerывание обучения** — не страшно: веса и история пишутся каждую эпоху, повторный запуск продолжит (`RESUME с эпохи N`).
- **Мало места** — `runs/` с весами ~2 МБ на ячейку; ZIP-пакет ~200 МБ (в основном `.git` и тест-артефакты репозитория).

## 10. Что почитать дальше

- `research/tinygpt_formula/README.md` — что уже сделано в исследовании (обучение, бенчмарк, открытые вопросы).
- `research/tinygpt_formula/scaling/SCALING_PLAN.md` — гипотезы H1-H4 и дизайн лестницы.
- `INSTRUCTIONS_TERMUX.md` (в корне ZIP-пакета) — пуш всего репозитория на GitHub с телефона.
- `colab/Scaling_Ladder_Hallucinations.ipynb` — тот же эксперимент на бесплатном Colab CPU/GPU.
