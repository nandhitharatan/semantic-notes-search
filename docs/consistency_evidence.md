# Consistency Evidence: PostgreSQL Outbox and Qdrant Recovery

This record demonstrates the temporary consistency gap created when Qdrant delivery
fails after PostgreSQL has committed the document, chunks, and `PENDING` job. The
retry worker later repairs the Qdrant index using the same chunk UUID point IDs.

## Before recovery

```text
doc_id | file_name                                                                                                | subject  | chunks | job_status
-------+----------------------------------------------------------------------------------------------------------+----------+--------+-----------
1      | dbms_notes.pdf.pdf                                                                                       | DBMS     | 6      | COMPLETED
2      | dbms_notes.pdf.pdf                                                                                       | FAILTEST | 6      | PENDING
3      | co answers updated (1).pdf                                                                               | Comp     | 77     | COMPLETED
4      | common  bus system.pdf                                                                                   | Comp     | 36     | COMPLETED
5      | Abraham-Silberschatz-Henry-F.-Korth-S.-Sudarshan-Database-System-Concepts-McGraw-Hill-Education-2019.pdf | DBMS     | 7883   | COMPLETED
6      | INDEX, JOINS.pdf                                                                                         | DBMS     | 20     | PENDING
7      | Integrity constraints and E.F Codd Rules.pdf                                                             | DBMS     | 14     | COMPLETED
8      | PROCEDURES AND CURSORS.pdf                                                                                | DBMS     | 10     | COMPLETED
9      | OOPS Using JAVA LAB MANUAL 2026-2027.pdf                                                                 | Java     | 253    | COMPLETED
10     | STLW_JAVA_UNIT_2.pdf                                                                                     | Java     | 159    | COMPLETED
Total chunks in PostgreSQL: 8464
points_count in Qdrant: 8438
```

The difference is 26 points, exactly matching the two PENDING documents: 6 chunks
for document 2 and 20 chunks for document 6.

## Failure test evidence

The manual Qdrant-outage test recorded these required facts before recovery:

```text
Chunks saved: 6
Ingestion job status: PENDING
PASS: PostgreSQL retained chunks and left the job PENDING while Qdrant was unavailable.
```

This proves that PostgreSQL is the durable source of truth: a Qdrant failure does not
lose the committed document or chunk rows.

## After recovery

```text
py database/retry_worker.py
No retryable PENDING jobs found.

py database/show_status.py
doc_id | file_name                                                                                                | subject  | chunks | job_status
-------+----------------------------------------------------------------------------------------------------------+----------+--------+-----------
1      | dbms_notes.pdf.pdf                                                                                       | DBMS     | 6      | COMPLETED
2      | dbms_notes.pdf.pdf                                                                                       | FAILTEST | 6      | COMPLETED
3      | co answers updated (1).pdf                                                                               | Comp     | 77     | COMPLETED
4      | common  bus system.pdf                                                                                   | Comp     | 36     | COMPLETED
5      | Abraham-Silberschatz-Henry-F.-Korth-S.-Sudarshan-Database-System-Concepts-McGraw-Hill-Education-2019.pdf | DBMS     | 7883   | COMPLETED
6      | INDEX, JOINS.pdf                                                                                         | DBMS     | 20     | COMPLETED
7      | Integrity constraints and E.F Codd Rules.pdf                                                             | DBMS     | 14     | COMPLETED
8      | PROCEDURES AND CURSORS.pdf                                                                                | DBMS     | 10     | COMPLETED
9      | OOPS Using JAVA LAB MANUAL 2026-2027.pdf                                                                 | Java     | 253    | COMPLETED
10     | STLW_JAVA_UNIT_2.pdf                                                                                     | Java     | 159    | COMPLETED
Total chunks in PostgreSQL: 8464
points_count in Qdrant: 8464
```

After recovery, every job is `COMPLETED` and the PostgreSQL chunk count matches the
Qdrant point count. The 26-point gap is closed.
