## Deploying the FastAPI Server on Heroku (cloud service provider)

### Step 1: Log in to Heroku from the command line
```
heroku login
```

### Step 2: Create a new Heroku app
```
heroku create your-fastapi-app-name
```

### Step 3: Initialize a Git repository
```
git init
```

### Step 4: Add and commit the files
```
git add .
git commit -m "Deploy FastAPI server"
```

### Step 5: Deploy the app
```
git push heroku master
```

### Step 6: Update Streamlit App
Once the API server is deployed, update the URL in the Streamlit app to point to the deployed server.
