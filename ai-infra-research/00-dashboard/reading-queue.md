---
tags: []
created: 2026-03-05
---

# 待读队列

## 高优先级
```dataview
TABLE subfield as "子领域", venue as "会议", url as "链接"
FROM #paper
WHERE status = "未读"
SORT created DESC
```

## 在读中
```dataview
TABLE subfield as "子领域", venue as "会议"
FROM #paper
WHERE status = "在读"
SORT created DESC
```