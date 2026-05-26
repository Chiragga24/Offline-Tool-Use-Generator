# ToolBench-Style Subset Proposal

This is a review draft for the representative ToolBench-style subset we plan to use for the offline synthetic conversation generator.

The goal is not to mirror all of ToolBench. The goal is to define a broad, realistic, reproducible subset that exercises:

- registry normalization
- tool graph construction
- constrained sampling
- stateful offline execution
- multi-step tool chains
- multi-domain tool chains
- conversation grounding
- diversity steering
- evaluation and repair

Important implementation assumption:

- All endpoints are offline mocks. Endpoints named `create_*`, `book_*`, `make_*`, or `add_*` simulate state transitions and return synthetic confirmations. They do not call real services, send notifications, create real calendar entries, make real reservations, or subscribe to live alert systems.

## Category Summary

Planned categories:

1. Finance
2. Sports
3. AI / ML
4. Entertainment
5. Travel
6. Gaming
7. Events
8. Food / Restaurants
9. Weather

Planned size:

- 5 endpoints per category
- 45 endpoints total

## 1. Finance

| Endpoint | Purpose | Key Inputs | Key Outputs |
|---|---|---|---|
| `finance/search_symbol` | Resolve a company, asset, or crypto name to a symbol. | `query`, `asset_type` | `symbol`, `asset_id`, `exchange` |
| `finance/get_quote` | Get the latest market quote. | `symbol` | `price`, `currency`, `change_pct` |
| `finance/get_company_news` | Get recent news for a company or asset. | `symbol`, `limit` | `article_id`, `headline`, `sentiment` |
| `finance/compare_assets` | Compare multiple assets. | `symbols[]` | `comparison_id`, `rankings` |
| `finance/create_price_alert` | Simulate creating a price movement alert preference. | `symbol`, `threshold`, `direction`, `delivery_method` | `alert_id`, `status` |

Example chain:

```text
finance/search_symbol
-> finance/get_quote
-> finance/get_company_news
-> finance/create_price_alert
```

## 2. Sports

| Endpoint | Purpose | Key Inputs | Key Outputs |
|---|---|---|---|
| `sports/search_team` | Resolve a team name to a team ID. | `query`, `league` | `team_id`, `team_name`, `league` |
| `sports/get_schedule` | Get upcoming games for a team. | `team_id`, `date_range` | `game_id`, `opponent`, `venue`, `start_time` |
| `sports/get_player_stats` | Get player statistics. | `player_name`, `team_id` | `player_id`, `stats` |
| `sports/compare_teams` | Compare two or more teams. | `team_ids[]` | `comparison_id`, `strengths` |
| `sports/get_game_odds` | Get betting or prediction odds for a game. | `game_id` | `odds_id`, `favorite`, `spread` |

Example chain:

```text
sports/search_team
-> sports/get_schedule
-> sports/get_game_odds
```

Cross-domain example:

```text
sports/search_team
-> sports/get_schedule
-> weather/get_forecast
-> events/create_calendar_event
```

## 3. AI / ML

| Endpoint | Purpose | Key Inputs | Key Outputs |
|---|---|---|---|
| `ai_ml/list_models` | Find models for a task or provider. | `task`, `provider` | `model_id`, `model_name`, `capabilities` |
| `ai_ml/get_model_details` | Inspect model metadata. | `model_id` | `context_window`, `pricing`, `latency_class` |
| `ai_ml/estimate_inference_cost` | Estimate model inference cost. | `model_id`, `tokens`, `requests` | `estimate_id`, `estimated_cost` |
| `ai_ml/create_eval_job` | Create an evaluation job for a model. | `model_id`, `dataset_name`, `metric` | `eval_job_id`, `status` |
| `ai_ml/get_eval_result` | Fetch the result of an evaluation job. | `eval_job_id` | `score`, `passed`, `summary` |

Planned AI / ML task values should intentionally connect to other categories:

```text
finance_sentiment
sports_prediction
travel_recommendation
game_recommendation
event_classification
restaurant_ranking
weather_risk_summary
entertainment_recommendation
```

Example chain:

```text
ai_ml/list_models
-> ai_ml/get_model_details
-> ai_ml/estimate_inference_cost
-> ai_ml/create_eval_job
-> ai_ml/get_eval_result
```

Cross-domain example:

```text
finance/get_company_news
-> ai_ml/list_models
-> ai_ml/create_eval_job
-> ai_ml/get_eval_result
```

## 4. Entertainment

| Endpoint | Purpose | Key Inputs | Key Outputs |
|---|---|---|---|
| `entertainment/search_movies` | Find movies or shows. | `query`, `genre` | `movie_id`, `title`, `release_year` |
| `entertainment/get_movie_details` | Get details for a movie or show. | `movie_id` | `rating`, `runtime`, `cast` |
| `entertainment/get_streaming_availability` | Find where a title can be streamed. | `movie_id`, `country` | `platforms`, `availability_id` |
| `entertainment/create_watchlist_item` | Add a movie or show to a watchlist. | `movie_id`, `user_id` | `watchlist_item_id`, `status` |
| `entertainment/search_live_shows` | Find live entertainment in a city. | `city`, `date`, `genre` | `show_id`, `venue`, `start_time` |

Example chain:

```text
entertainment/search_movies
-> entertainment/get_movie_details
-> entertainment/get_streaming_availability
-> entertainment/create_watchlist_item
```

Cross-domain example:

```text
entertainment/search_live_shows
-> food/search_restaurants
-> food/check_availability
-> food/make_reservation
```

## 5. Travel

| Endpoint | Purpose | Key Inputs | Key Outputs |
|---|---|---|---|
| `travel/search_flights` | Find flights for a route and date. | `origin`, `destination`, `date` | `flight_id`, `airline`, `price` |
| `travel/search_hotels` | Find hotels in a city. | `city`, `check_in`, `max_price` | `hotel_id`, `hotel_name`, `nightly_price` |
| `travel/search_travel_deals` | Find discounts or package deals for a destination. | `destination`, `date`, `deal_type` | `deal_id`, `discount_pct`, `eligible_items` |
| `travel/get_hotel_details` | Get hotel details. | `hotel_id` | `amenities`, `rating`, `address` |
| `travel/book_itinerary` | Simulate booking selected flight and hotel options. | `flight_id`, `hotel_id`, `traveler_name`, `deal_id` | `booking_id`, `status` |

Example chain:

```text
travel/search_flights
-> travel/search_hotels
-> travel/search_travel_deals
-> travel/get_hotel_details
-> travel/book_itinerary
```

Cross-domain example:

```text
travel/search_flights
-> weather/get_forecast
-> events/search_events
-> food/search_restaurants
```

## 6. Gaming

| Endpoint | Purpose | Key Inputs | Key Outputs |
|---|---|---|---|
| `gaming/search_games` | Find games by query, platform, genre, store, or price type. | `query`, `platform`, `genre`, `store`, `price_type`, `max_price` | `game_id`, `title`, `platforms`, `store`, `price`, `is_free` |
| `gaming/get_game_details` | Get game metadata. | `game_id` | `rating`, `developer`, `release_date` |
| `gaming/get_player_profile` | Get a player's gaming profile. | `username`, `platform` | `player_id`, `rank`, `recent_games` |
| `gaming/recommend_games` | Recommend games using game and player context. | `game_id`, `player_id` | `recommendation_id`, `games[]` |
| `gaming/get_tournament_schedule` | Find tournaments for a game. | `game_id`, `region` | `tournament_id`, `start_time`, `venue` |

Planned gaming values:

```text
platform = pc | xbox | playstation | switch | mobile
store = steam | epic | xbox_store | playstation_store | nintendo_eshop | app_store
price_type = free | paid | subscription
```

Example chain:

```text
gaming/search_games
-> gaming/get_game_details
-> gaming/recommend_games
```

Cross-domain example:

```text
gaming/search_games
-> gaming/get_tournament_schedule
-> events/create_calendar_event
```

## 7. Events

| Endpoint | Purpose | Key Inputs | Key Outputs |
|---|---|---|---|
| `events/search_events` | Find events in a city or category. | `city`, `date`, `category` | `event_id`, `name`, `venue_id` |
| `events/get_event_details` | Get details for an event. | `event_id` | `venue`, `start_time`, `ticket_required` |
| `events/check_ticket_availability` | Check ticket availability. | `event_id`, `quantity` | `ticket_offer_id`, `price`, `available` |
| `events/book_tickets` | Book event tickets. | `ticket_offer_id`, `attendee_name` | `ticket_booking_id`, `status` |
| `events/create_calendar_event` | Simulate adding a public event, meeting, reservation, game, reminder, or tournament to a calendar. | `title`, `start_time`, `location`, `event_type`, `attendees` | `calendar_event_id`, `status` |

Planned event type values:

```text
public_event
meeting
reservation
sports_game
travel_reminder
tournament
watch_party
```

Example chain:

```text
events/search_events
-> events/get_event_details
-> events/check_ticket_availability
-> events/book_tickets
-> events/create_calendar_event
```

## 8. Food / Restaurants

| Endpoint | Purpose | Key Inputs | Key Outputs |
|---|---|---|---|
| `food/search_restaurants` | Find restaurants by location and cuisine. | `city`, `cuisine`, `party_size` | `restaurant_id`, `name`, `rating` |
| `food/get_menu` | Get restaurant menu information. | `restaurant_id` | `menu_id`, `popular_items`, `price_range` |
| `food/check_availability` | Check reservation slots. | `restaurant_id`, `date`, `time`, `party_size` | `slot_id`, `available_time` |
| `food/make_reservation` | Reserve a table. | `slot_id`, `customer_name` | `reservation_id`, `status` |
| `food/get_dietary_options` | Check dietary matches. | `restaurant_id`, `dietary_need` | `matching_items`, `confidence` |

Example chain:

```text
food/search_restaurants
-> food/get_menu
-> food/check_availability
-> food/make_reservation
```

## 9. Weather

| Endpoint | Purpose | Key Inputs | Key Outputs |
|---|---|---|---|
| `weather/get_forecast` | Get weather forecast by city and date. | `city`, `date` | `forecast_id`, `condition`, `temperature` |
| `weather/get_active_alerts` | Look up active or forecasted weather alerts for a city/date. | `city`, `date` | `alert_id`, `severity`, `description` |
| `weather/compare_destination_weather` | Compare weather across multiple cities. | `cities[]`, `date` | `comparison_id`, `best_city` |
| `weather/get_hourly_forecast` | Get hourly forecast details. | `city`, `date` | `hourly[]`, `rain_probability` |
| `weather/recommend_outdoor_window` | Recommend the best time window for an outdoor activity. | `city`, `date`, `activity` | `window_id`, `start_time`, `end_time` |

Example chain:

```text
weather/get_forecast
-> weather/get_hourly_forecast
-> weather/recommend_outdoor_window
```

## Important Connection Fields

These fields should be designed intentionally because they allow the graph and executor to create grounded chains:

```text
city
date
start_time
venue
team_id
game_id
event_id
restaurant_id
slot_id
flight_id
hotel_id
booking_id
model_id
eval_job_id
symbol
movie_id
player_id
tournament_id
forecast_id
deal_id
alert_id
```

## Rich Multi-Domain Chains To Support

### Travel + Weather + Events + Food

```text
travel/search_flights
-> weather/get_forecast
-> events/search_events
-> food/search_restaurants
-> food/check_availability
-> food/make_reservation
```

### Sports + Weather + Events

```text
sports/search_team
-> sports/get_schedule
-> weather/get_forecast
-> events/create_calendar_event
```

### Entertainment + Events + Food

```text
entertainment/search_live_shows
-> events/search_events
-> food/search_restaurants
-> food/make_reservation
```

### AI / ML + Finance

```text
finance/search_symbol
-> finance/get_company_news
-> ai_ml/list_models
-> ai_ml/create_eval_job
-> ai_ml/get_eval_result
```

### Gaming + Events

```text
gaming/search_games
-> gaming/get_tournament_schedule
-> events/create_calendar_event
```
