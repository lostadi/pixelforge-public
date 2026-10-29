"""Read-only, conversation-scoped access to durable tool receipts."""
import json
import re

from agent import AppError, clean


def _number(value, name, low, high):
    if not isinstance(value, str) or not re.fullmatch(r'[0-9]{1,18}', value):
        raise AppError('Invalid saved-result ' + name)
    number = int(value)
    if not low <= number <= high:
        raise AppError('Invalid saved-result ' + name)
    return number


def dispatch(agent, args):
    argv = args.get('argv', [])
    if not isinstance(argv, list) or not 2 <= len(argv) <= 5 or argv[0] != '@results':
        raise AppError('Use @results list [OFFSET [LIMIT]] or @results read EVENT [OFFSET [LIMIT]]')
    store, sid = agent.store, agent.sid
    if agent.cancelled.is_set():
        raise AppError('Saved-result request was cancelled')
    if argv[1] == 'list' and len(argv) <= 4:
        offset = _number(argv[2], 'offset', 0, 1000000000) if len(argv) > 2 else 0
        limit = _number(argv[3], 'limit', 1, 50) if len(argv) > 3 else 10
        with store.lock:
            rows = store.db.execute("SELECT seq,data FROM events WHERE session=? AND kind='tool_result' "
                                    "ORDER BY seq DESC LIMIT ? OFFSET ?", (sid, limit + 1, offset)).fetchall()
        results = []
        for row in rows[:limit]:
            data = clean(json.loads(row['data']))
            result = data.get('result', {})
            preview = json.dumps(result, ensure_ascii=False, separators=(',', ':'))
            results.append({'event': row['seq'], 'success': data.get('success'),
                            'preview': preview[:240],
                            'read_argv': ['@results', 'read', str(row['seq']), '0', '4000']})
        page = {'results': results, 'offset': offset, 'next_offset': offset + len(results),
                'more': len(rows) > len(results)}
        # Leave room for the model-facing success/receipt envelope. A result
        # index must itself remain readable without recursive preview clipping.
        while len(json.dumps(page, ensure_ascii=False)) > 5000 and len(results) > 1:
            results.pop()
            page.update(next_offset=offset + len(results), more=True)
        return page
    if argv[1] == 'read' and len(argv) >= 3:
        event = _number(argv[2], 'event', 1, 2**63 - 1)
        offset = _number(argv[3], 'offset', 0, 1000000000) if len(argv) > 3 else 0
        limit = _number(argv[4], 'limit', 1, 12000) if len(argv) > 4 else 4000
        with store.lock:
            row = store.db.execute("SELECT data FROM events WHERE session=? AND kind='tool_result' AND seq=?",
                                   (sid, event)).fetchone()
        if row is None:
            raise AppError('No saved tool result with that event in this conversation')
        data = clean(json.loads(row['data']))
        content = json.dumps(data.get('result', {}), ensure_ascii=False, separators=(',', ':'))
        if offset > len(content):
            raise AppError('Saved-result offset is past the end')
        def page(size):
            end = offset + size
            return {'event': event, 'success': data.get('success'), 'content': content[offset:end],
                    'offset': offset, 'next_offset': end, 'more': end < len(content), 'total_chars': len(content)}
        # Quoted/backslash-heavy JSON expands again inside the result envelope.
        # Advance only by the content actually delivered, never by the request's
        # upper limit, so all later pages remain lossless.
        low, high = 0, min(limit, len(content) - offset)
        while low < high:
            middle = (low + high + 1) // 2
            if len(json.dumps(page(middle), ensure_ascii=False)) <= 5000:
                low = middle
            else:
                high = middle - 1
        return page(low)
    raise AppError('Use @results list [OFFSET [LIMIT]] or @results read EVENT [OFFSET [LIMIT]]')
