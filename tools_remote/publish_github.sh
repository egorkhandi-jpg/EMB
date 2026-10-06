#!/usr/bin/env bash
# Публикация на GitHub (origin) фильтрованной копии истории — без CLAUDE.md (решение Егора 2026-10-06:
# CLAUDE.md и сведения о других проектах на этой машине не должны быть видны в репозитории).
# Шаблоны запрещённых строк — в локальном файле tools_remote/.publish_patterns (вне git): строки
# «i:<шаблон>» — без учёта регистра, «c:<шаблон>» — с учётом регистра. Без файла публикация не идёт.
#
# Локальная история не трогается: хэши в отчётах и git_commit в results_*.json остаются верными локально.
# Фильтр детерминирован (те же деревья, авторы, даты, сообщения) — повторный запуск даёт те же хэши,
# поэтому публикация идёт обычным push, без --force.
# Соответствие «локальный хэш → опубликованный» пишется в docs/COMMIT_MAP.md и коммитится локально
# (таблица покрывает все коммиты, кроме самого коммита с таблицей).
#
# Использование (Git Bash):  tools_remote/publish_github.sh [--dry-run]
#   --dry-run — отфильтровать, проверить и показать, что ушло бы, без push.
set -euo pipefail
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1
HERE=$(cd "$(dirname "$0")/.." && pwd)
URL=$(git -C "$HERE" remote get-url origin)
WORK="$(cygpath -u "${TEMP:-$HOME}")/emb_publish_$$"
MAP="$HERE/docs/COMMIT_MAP.md"
trap 'rm -rf "$WORK"' EXIT

filter() {
  rm -rf "$WORK"; mkdir -p "$WORK"
  git clone --quiet --mirror "$HERE" "$WORK/pub.git"
  cd "$WORK/pub.git"
  SRC=$(git rev-parse master)            # что попало в копию (параллельные коммиты после клона — в следующий раз)
  # публикуется только master и метки: прочие ссылки (refs/remotes/vast/…) удаляются
  git for-each-ref --format='%(refname)' | grep -v -e '^refs/heads/master$' -e '^refs/tags/' | while read -r r; do git update-ref -d "$r"; done
  FILTER_BRANCH_SQUELCH_WARNING=1 git filter-branch -f --index-filter 'git rm --cached --ignore-unmatch -q CLAUDE.md' \
    --tag-name-filter cat -- master --tags >/dev/null
  git for-each-ref --format='%(refname)' refs/original | while read -r r; do git update-ref -d "$r"; done
  cd "$HERE"
}

check() {
  local P="$WORK/pub.git" PAT="$HERE/tools_remote/.publish_patterns"
  [ -s "$PAT" ] || { echo "ОШИБКА: нет $PAT — шаблоны запрещённых строк"; exit 1; }
  [ -z "$(git -C "$P" log --all --format=%h -- CLAUDE.md)" ] || { echo "ОШИБКА: CLAUDE.md остался в истории копии"; exit 1; }
  local revs; revs=$(git -C "$P" rev-list --all)
  while IFS= read -r line; do
    [ -n "$line" ] || continue
    local mode=${line%%:*} pat=${line#*:} opt=""; [ "$mode" = "i" ] && opt="-i"
    if git -C "$P" grep -q $opt -F -e "$pat" $revs -- 2>/dev/null; then echo "ОШИБКА: запрещённая строка в файлах копии (шаблон $mode)"; exit 1; fi
    if [ -n "$(git -C "$P" log --all --format=%h -F ${opt:+--regexp-ignore-case} --grep="$pat")" ]; then echo "ОШИБКА: запрещённая строка в сообщениях коммитов (шаблон $mode)"; exit 1; fi
  done < "$PAT"
  local a b; a=$(git -C "$HERE" rev-list --count "$SRC"); b=$(git -C "$P" rev-list --count master)
  [ "$a" = "$b" ] || { echo "ОШИБКА: число коммитов разное ($a против $b)"; exit 1; }
}

write_map() {
  local P="$WORK/pub.git" tmp="$WORK/map.md"
  {
    echo "# Соответствие коммитов: локальный репозиторий → опубликованная копия (GitHub)"
    echo
    echo "Опубликованная копия — та же история без файла CLAUDE.md (\`tools_remote/publish_github.sh\`)."
    echo "Хэши в отчётах и поле git_commit в results_*.json — локальные; здесь — их опубликованные пары."
    echo "Таблица покрывает все коммиты, кроме самого коммита с таблицей."
    echo
    echo "| Локальный | Опубликованный | Дата | Сообщение |"
    echo "| --- | --- | --- | --- |"
    paste -d'|' <(git -C "$HERE" rev-list --topo-order --reverse "$SRC") <(git -C "$P" rev-list --topo-order --reverse master) |
    while IFS='|' read -r o n; do
      so=$(git -C "$HERE" log -1 --format=%s "$o"); sn=$(git -C "$P" log -1 --format=%s "$n")
      [ "$so" = "$sn" ] || { echo "ОШИБКА: порядок коммитов не совпал ($o / $n)" >&2; exit 1; }
      printf '| %s | %s | %s | %s |\n' "${o:0:7}" "${n:0:7}" "$(git -C "$HERE" log -1 --format=%ad --date=short "$o")" "$(echo "$so" | cut -c1-90 | sed 's/|/\\|/g')"
    done
    echo
    echo "Метки (в копии указывают на опубликованные пары): $(git -C "$P" tag | tr '\n' ' ')"
  } > "$tmp"
  echo "$tmp"
}

filter; check
tmp=$(write_map)
if ! cmp -s "$tmp" "$MAP"; then
  # таблица изменилась — локальный коммит с ней, затем фильтр заново (прежние хэши копии не меняются)
  cp "$tmp" "$MAP"
  git -C "$HERE" add docs/COMMIT_MAP.md
  git -C "$HERE" commit -q -m "COMMIT_MAP: соответствие хэшей локальной истории и опубликованной копии (без CLAUDE.md)"
  filter; check
fi
echo "копия готова: $(git -C "$WORK/pub.git" rev-list --count master) коммитов, меток $(git -C "$WORK/pub.git" tag | wc -l); CLAUDE.md и запрещённых строк в истории нет"
echo "вершина: локально ${SRC:0:7} → в копии $(git -C "$WORK/pub.git" rev-parse --short master)"
if [ $DRY = 1 ]; then
  echo "--dry-run: push не выполнялся (адрес $URL)"; exit 0
fi
git -C "$WORK/pub.git" push "$URL" master --tags
