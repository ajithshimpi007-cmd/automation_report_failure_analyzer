
from behave import given, when, then
from selenium import webdriver
from selenium.webdriver.common.by import By
import time

@given('the user is on the login page')
def step_impl(context):
    context.driver = webdriver.Chrome()
    context.driver.get("https://www.facebook.com/")
    
    
@when('the user enters valid credentials')
def step_impl(context):
    context.driver.find_element(By.ID, "email").send_keys("ajithshimpi007@gmail.com")
    context.driver.find_element(By.ID, "pass").send_keys("password123")

@when('clicks the login button')
def step_impl(context):
    context.driver.find_element(By.NAME, "login").click()
    time.sleep(5)

@then('the user should be redirected to the homepage')
def step_impl(context):
    assert "Facebook" in context.driver.title
    context.driver.quit()

