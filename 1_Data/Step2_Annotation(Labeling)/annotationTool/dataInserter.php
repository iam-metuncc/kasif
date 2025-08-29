<?php

require 'config.php'; // Include database connection and session configuratio

$parentFolder = 'webpages'; // The main folder where all page folders are stored


//Get all folder names (each folder represents a web page)
$folders = scandir($parentFolder);
// echo "Remaining folders: \n";
// print_r($folders);


$numberOfPages = count($folders);

//we start from index 2 because first two indices are reserved for current and parent directories
$i = 2;


if ($numberOfPages > 2) {

    while ($i < $numberOfPages) {

        //Current folder to be inserted should be the first in the array
        $pageName = $folders[$i];
        $i = $i + 1;

        // Build the full path to the current page’s folder
        $currentDirectory = $parentFolder . '/' . $pageName . '/';

        $images = glob($currentDirectory . "*.png");
        $numberOfImages = count($images);
        
        // Initialize table counter
        $j = 1;
        while ($j <= $numberOfImages) {

            // Create a table name like "table1", "table2", etc.
            $tableName = 'table' . $j;

            //Preparing the query string to insert a record into the annotations tabl
            $queryString = 'INSERT INTO annotations (pageName, tableNumber) VALUES ("' . $pageName . '","' . $tableName . '")';

            // Execute the query
            $query = mysqli_query($con, $queryString);

            if (!$query) {
                echo "Error (updating records): " . mysqli_error($con);
            }
            // If the query is successful, print the inserted values
            else{
                //nl2br is used so that each line is printed in a new line in browsers too and not just consoles
                echo nl2br("(Inserted) pageName: " . $pageName . "      tableName: " . $tableName . "\n");
            }


            $j = $j + 1;
        }
    }
} else {
    echo "No folders (pages) remaining in the /webpages/ directory to insert.";
}
