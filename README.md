# SpamCheck
Application is designed to check string is spam. Algorith using Naive Bayers Classifier. 

## Endpoints
### POST `/email/spam/check`

#### Body Parameters
- `data_to_check` (string): Data to predice.
#### Response
Returns a boolean. If return 
- `0` : Given string is not spam.
- `1` : Given string is spam.

## Usage
You can start the server by running the uvicorn server by calling a command: 'uvicorn main:app'. The server will start on `0.0.0.0:3100`.
Alternative you can run a application as a Docker
