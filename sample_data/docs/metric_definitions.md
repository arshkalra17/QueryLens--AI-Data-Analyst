# Metric Definitions

Owner: Analytics team. These are the official definitions used in every dashboard and
report. When a question uses one of these terms, apply the definition exactly as written.
Each definition lists the dataset it applies to and, where possible, the SQL filter or
expression that implements it.

## High-value transaction

Applies to: retail_transactions.csv

A retail transaction whose order total is greater than 10,000 rupees. High-value
transactions are tracked separately because they are reviewed by the fraud team and
qualify the customer for the loyalty programme.

SQL filter: `"total" > 10000`

## Low-rated transaction

Applies to: retail_transactions.csv

A transaction where the customer gave a rating of 2 or lower on the 1–5 scale.
Low-rated transactions trigger a follow-up call from customer support within 48 hours.

SQL filter: `"rating" <= 2`

## Satisfied purchase

Applies to: retail_transactions.csv

A transaction rated 4 or 5. The satisfaction rate is the share of transactions that are
satisfied purchases; the company target is 60%.

SQL filter: `"rating" >= 4`

## Bulk order

Applies to: sales_data.csv

A sale of 20 or more units in a single order line. Bulk orders receive volume pricing
and are excluded from the sales-rep individual bonus calculation.

SQL filter: `"units_sold" >= 20`

## Premium product sale

Applies to: sales_data.csv

A sale whose unit price is at least 1,000. Premium sales are reported separately in the
quarterly business review because their margins differ from the core range.

SQL filter: `"unit_price" >= 1000`

## Gross margin

Applies to: sales_data.csv

Gross margin is revenue minus cost, summed over the period. It is reported in currency,
not as a percentage. Do not confuse it with the margin rate.

SQL expression: `SUM("revenue") - SUM("cost")`

## Margin rate

Applies to: sales_data.csv

Margin rate is gross margin divided by revenue, expressed as a percentage. A margin rate
below 30% on a product category is flagged for pricing review.

SQL expression: `100.0 * (SUM("revenue") - SUM("cost")) / SUM("revenue")`

## High-bounce day

Applies to: web_traffic.csv

A day on which the bounce rate was above 60%. Three consecutive high-bounce days open an
incident with the web team.

SQL filter: `"bounce_rate" > 60`

## Strong conversion day

Applies to: web_traffic.csv

A day with a conversion rate of 4% or more. Marketing uses strong conversion days to
evaluate campaign creative.

SQL filter: `"conversion_rate" >= 4`

## Cost per session

Applies to: web_traffic.csv

Ad spend divided by sessions over the same period. It is the primary efficiency metric for
paid acquisition; the target is below 6.

SQL expression: `SUM("ad_spend") / SUM("sessions")`

## Early subscriber

Applies to: customers-1000_2.csv

A customer whose subscription date is before 1 January 2021. Early subscribers keep the
original pricing plan.

SQL filter: `"Subscription Date" < '2021-01-01'`
