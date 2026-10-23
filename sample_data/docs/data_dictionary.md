# Data Dictionary

This dictionary describes every dataset shipped with QueryLens: where it comes from, how
often it is refreshed, and what each column means.

## sales_data.csv

Order-line sales of electronics accessories from the B2B sales system. One row is one
order line. Refreshed nightly at 02:00 UTC.

| Column | Type | Description |
|---|---|---|
| date | date | Date the order was booked (not shipped). |
| region | text | Sales region: North, South, East, West or Central. |
| product_category | text | One of Charger, Headset, Keyboard, Laptop, Monitor, Mouse, Phone, Tablet. |
| units_sold | integer | Number of units on the order line. |
| unit_price | decimal | Price per unit after discount, in USD. |
| revenue | decimal | units_sold × unit_price, in USD. |
| cost | decimal | Landed cost of the units, in USD. |
| channel | text | Route to market: Online, Retail, Partner or Distributor. |
| customer_id | text | Customer account code (CUS####). |
| sales_rep | text | Account owner. Blank when the order came through self-service. |

### Known data quality issues

Some rows have a blank sales_rep; these are self-service orders and are expected. A small
number of order lines have revenue far above the usual range because enterprise deals
are booked as a single line.

## retail_transactions.csv

Point-of-sale transactions from the apparel stores in seven Indian cities. One row is one
transaction. Amounts are in Indian rupees (INR).

| Column | Type | Description |
|---|---|---|
| transaction_id | text | Unique transaction reference (TXN######). |
| date | timestamp | Time of purchase, store local time. |
| customer_id | text | Loyalty-card customer code (CUST#####). |
| customer_name | text | Display name on the loyalty card. |
| city | text | Store city: Mumbai, Delhi, Pune, Kolkata, Chennai, Hyderabad or Bengaluru. |
| item | text | Product line, e.g. Belt, Shoes, Jacket. |
| quantity | integer | Units purchased (1–4). |
| unit_price | decimal | Shelf price per unit in INR. |
| total | decimal | Amount charged for the transaction in INR (the order total). |
| payment_method | text | Cash, Card, UPI, Wallet or NetBanking. |
| rating | integer | Post-purchase satisfaction survey, 1 (very poor) to 5 (excellent). |

## web_traffic.csv

Daily website analytics for the online store, one row per day, exported from the web
analytics platform.

| Column | Type | Description |
|---|---|---|
| date | date | Calendar day (UTC). |
| pageviews | integer | Total page views that day. |
| sessions | integer | Visits; a session ends after 30 minutes of inactivity. |
| conversion_rate | decimal | Percentage of sessions that ended in a purchase. |
| bounce_rate | decimal | Percentage of sessions with a single page view. |
| ad_spend | integer | Paid-media spend that day, in USD. |

## customers-1000_2.csv

Customer master records for the subscription business.

| Column | Type | Description |
|---|---|---|
| Customer Id | text | Unique customer identifier. |
| Country | text | Billing country. |
| Subscription Date | date | Date the customer first subscribed. |
| Email | text | Contact email. Personal data: do not export. |
| Phone 1 / Phone 2 | text | Contact numbers. Personal data: do not export. |
