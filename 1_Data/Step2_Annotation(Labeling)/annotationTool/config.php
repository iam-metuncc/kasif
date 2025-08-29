<?php

ob_start();
session_start();

// Set the default timezone for the application
$timeZone = date_default_timezone_set("Asia/Nicosia");


// Establish a connection to the MySQL database
// Parameters: hostname, username, password, database name
$con = mysqli_connect("localhost", "root", "", "annotationtool");

// Check if the connection failed and display the error code if so
if(mysqli_connect_errno()){
    echo "Failed to connect" . mysqli_connect_errno();
}


?>