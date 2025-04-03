library(shiny)
library(mongolite)
library(ggplot2)
library(dplyr)
library(caret)
library(xgboost)
library(DT)
library(shinydashboard)
library(plotly)

# UI
ui <- dashboardPage(
  dashboardHeader(title = "Tesla Stock Analysis"),
  dashboardSidebar(
    sidebarMenu(
      menuItem("Load Data", tabName = "data", icon = icon("database")),
      menuItem("EDA", tabName = "eda", icon = icon("chart-bar")),
      menuItem("Train Model", tabName = "model", icon = icon("rocket"))
    )
  ),
  dashboardBody(
    tabItems(
      tabItem(tabName = "data",
              actionButton("load_data", "📥 Load Data"),
              DTOutput("data_table")
      ),
      tabItem(tabName = "eda",
              fluidRow(
                column(6, plotlyOutput("eda_plot_1")),
                column(6, plotlyOutput("eda_plot_2"))
              )
      ),
      tabItem(tabName = "model",
              actionButton("train_model", "🚀 Train Model"),
              verbatimTextOutput("rmse_output"),
              verbatimTextOutput("accuracy_output"),
              plotOutput("prediction_plot"),
              verbatimTextOutput("next_day_prediction")
      )
    )
  )
)

# Server
server <- function(input, output) {
  df <- reactiveVal(NULL)
  xgb_model <- reactiveVal(NULL)
  
  observeEvent(input$load_data, {
    mongo_conn <- mongo(collection = "r3", db = "project", url = "mongodb://localhost:27017")
    data <- mongo_conn$find('{}')
    
    data$Date <- as.Date(data$Date, format="%m/%d/%Y")
    df(data)
    output$data_table <- renderDT({ datatable(data) })
  })
  
  output$eda_plot_1 <- renderPlotly({
    if (is.null(df())) return()
    ggplotly(ggplot(df(), aes(x = Date, y = Close)) +
               geom_line(color = "blue") + theme_minimal() +
               labs(title = "Closing Price Over Time", x = "Date", y = "Close Price"))
  })
  
  output$eda_plot_2 <- renderPlotly({
    if (is.null(df())) return()
    ggplotly(ggplot(df(), aes(x = Volume, y = Close)) +
               geom_point(alpha = 0.5) + theme_minimal() +
               labs(title = "Volume vs. Closing Price", x = "Volume", y = "Close Price"))
  })
  
  observeEvent(input$train_model, {
    if (is.null(df())) {
      output$rmse_output <- renderPrint({ "❌ No data available! Load the data first." })
      return()
    }
    
    data <- df()
    data <- data %>% arrange(Date)
    data$Close_next <- lead(data$Close, 1)
    data <- na.omit(data)
    
    feature_cols <- c("Open", "High", "Low", "Close", "Volume")
    
    set.seed(123)
    train_index <- createDataPartition(data$Close_next, p = 0.8, list = FALSE)
    train_data <- data[train_index, ]
    test_data  <- data[-train_index, ]
    
    train_matrix <- xgb.DMatrix(data = as.matrix(train_data[, feature_cols]), label = train_data$Close_next)
    test_matrix  <- xgb.DMatrix(data = as.matrix(test_data[, feature_cols]), label = test_data$Close_next)
    
    param_grid <- expand.grid(
      max_depth = c(3, 6, 9),
      eta = c(0.01, 0.1, 0.3),
      nrounds = c(50, 100, 200)
    )
    
    best_rmse <- Inf
    best_params <- NULL
    
    for (i in 1:nrow(param_grid)) {
      params <- list(
        objective = "reg:squarederror",
        eval_metric = "rmse",
        max_depth = param_grid$max_depth[i],
        eta = param_grid$eta[i]
      )
      model <- xgb.train(params = params, data = train_matrix, nrounds = param_grid$nrounds[i])
      predictions <- predict(model, test_matrix)
      rmse <- sqrt(mean((predictions - test_data$Close_next)^2))
      
      if (rmse < best_rmse) {
        best_rmse <- rmse
        best_params <- params
        xgb_model(model)
      }
    }
    
    output$rmse_output <- renderPrint({ paste("✅ Best RMSE:", round(best_rmse, 2)) })
    
    output$prediction_plot <- renderPlot({
      best_model <- xgb_model()
      predictions <- predict(best_model, test_matrix)
      
      plot(test_data$Date, test_data$Close_next, type = "l", col = "blue", lwd = 2, xlab = "Date", ylab = "Close Price", main = "Actual vs Predicted")
      lines(test_data$Date, predictions, col = "red", lwd = 2)
      legend("topright", legend = c("Actual", "Predicted"), col = c("blue", "red"), lty = 1, lwd = 2)
    })
    
    # Compute accuracy (R-squared)
    best_model <- xgb_model()
    predictions <- predict(best_model, test_matrix)
    ss_total <- sum((test_data$Close_next - mean(test_data$Close_next))^2)
    ss_residual <- sum((test_data$Close_next - predictions)^2)
    r_squared <- 1 - (ss_residual / ss_total)
    
    output$accuracy_output <- renderPrint({ paste("🎯 Model Accuracy (R-squared):", round(r_squared, 4)) })
    
    # Predict next day's closing price
    latest_data <- tail(data, 1)[, feature_cols]
    latest_matrix <- xgb.DMatrix(data = as.matrix(latest_data))
    next_day_price <- predict(xgb_model(), latest_matrix)
    
    output$next_day_prediction <- renderPrint({ paste("📈 Predicted Next Day Close Price:", round(next_day_price, 2)) })
  })
}

shinyApp(ui = ui, server = server)
