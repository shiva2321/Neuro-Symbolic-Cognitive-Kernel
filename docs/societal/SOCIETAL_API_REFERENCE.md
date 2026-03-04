# SOCIETAL_API_REFERENCE.md: V5 Dashboard API

## Base URL
`http://localhost:8080`

## Endpoints

### 1. System Status
`GET /api/v5/status`
Returns high-level telemetry of the Knowledge World.
- **Response**:
  ```json
  {
    "epoch": 124,
    "population": 2500,
    "domain_count": 12,
    "hnsw_nodes": 2612,
    "zipf_alpha": 1.02
  }
  ```

### 2. World Hierarchy
`GET /api/v5/world`
Returns the recursive structure of all active domains and neighborhoods.
- **Response**:
  ```json
  [
    {
      "id": "physics_domain_0",
      "level": "State",
      "neighborhoods": [
        {"id": "quantum_mechanics_nh", "member_count": 42},
        {"id": "thermodynamics_nh", "member_count": 35}
      ]
    }
  ]
  ```

### 3. Domain Details
`GET /api/v5/domain/{domain_id}`
Fetches detailed state and health history for a specific domain.

### 4. Neighborhood Details
`GET /api/v5/neighborhood/{nh_id}`
Returns the anchor hypervector and list of member concept IDs.

### 5. Concept (Person) Details
`GET /api/v5/person/{concept_id}`
Returns the full properties and current social energy of a single hypervector.

### 6. World Advancment
`POST /api/v5/tick`
Triggers a world epoch advance, running percolation monitors and TDA health checks.

### 7. Concept Ingestion
`POST /api/v5/ingest`
- **Body**:
  ```json
  {
    "concept_id": "new_fact_42",
    "text": "The speed of light is constant in a vacuum."
  }
  ```
- **Action**: Extracts features via VSA, assigns a `LivingHyperVector`, and finds its semantic social circle.
