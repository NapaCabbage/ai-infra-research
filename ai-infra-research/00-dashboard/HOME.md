# AI Infra Research Dashboard

## 最近阅读的论文
```dataview
TABLE
  subfield as "子领域",
  venue as "会议",
  rating as "评分",
  status as "状态"
FROM #paper
SORT created DESC
LIMIT 10
```

## 待读队列
```dataview
TABLE subfield as "子领域", venue as "会议"
FROM #paper
WHERE status = "未读"
SORT created DESC
```

## 最近更新的概念
```dataview
LIST
FROM #concept
SORT file.mtime DESC
LIMIT 10
```

## 追踪的研究者
```dataview
TABLE institution as "机构", subfields as "方向"
FROM #researcher
SORT title ASC
```

## 统计
- 论文总数：`$= dv.pages("#paper").length`
- 已读：`$= dv.pages("#paper").where(p => p.status == "已读").length`
- 概念数：`$= dv.pages("#concept").length`