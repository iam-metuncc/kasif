<?php

require 'config.php';

// ----------------------------------------------------------------------
// Store the annotator number (passed via GET request) into the session
// Example: index.php?annotator=1

$_SESSION['annotator'] = $_GET['annotator'];
// echo "\nAnnotator is: \n";

// Depending on annotator number, load folders (web pages to be annotated)
// Currently, annotators 1–3 work with the "webpages" directory

if ($_SESSION['annotator'] == 1 || $_SESSION['annotator'] == 2 || $_SESSION['annotator'] == 3)
    //Get all folder names (each folder represents a web page)
    $folders = scandir("webpages");
elseif($_SESSION['annotator'] == 4 || $_SESSION['annotator'] == 5 || $_SESSION['annotator'] == 6)
    $folders = scandir("webpages1");  
   
$numberOfPages = count($folders);
// echo "\nNumber of folders (including current+parent): \n";
// print_r($numberOfPages);
//----------------------------------------------------------------------------------------------------
// Each annotator has a checkpoint file that stores progress
// For example: annotator1.txt, annotator2.txt, etc.

$annotatorFile = 'annotator' . $_SESSION['annotator'] . '.txt';
$_SESSION['annotatorFile'] = $annotatorFile;

// Open annotator checkpoint file and read the saved progress
$file = fopen($annotatorFile, "r") or die("Annotator file:" . $annotatorFile . " could not be opened");
$dataLine = fgets($file);

// Split checkpoint data into pageName and tableNumber
$checkPoints = explode(" ", $dataLine);

//----------------------------------------------------------------------------------------------------//

//If we have any folders in the webpages directory 
//(we start from 2 because first two elements are reserved for parent and current directories)
if($numberOfPages > 2){

    // If user clicked "nextPage" button → go to the next page
    if(isset($_GET['nextPage'])){
        $currentPageIndex = array_search($_SESSION['pageName'], $folders);
        
        $nextPageIndex = $currentPageIndex + 1;

        // If there are still pages left → load next page
        if($nextPageIndex < $numberOfPages){
            $_SESSION['pageName'] = $folders[$nextPageIndex];
            $_SESSION['tableNumber'] = '1';
            $_SESSION['tableName'] = 'table1';

            // Save progress to annotator checkpoint file
            file_put_contents($_SESSION['annotatorFile'], $_SESSION['pageName'] . ' ' . $_SESSION['tableNumber']);
        }
        // If no more pages → end annotation session
        else{
            echo "No more pages to annotate";
            file_put_contents($_SESSION['annotatorFile'], 'end end');
            exit();
        }

    }
    // If checkpoint file says "end end" → no more pages left
    elseif($checkPoints[0] == 'end' && $checkPoints[1] == 'end'){
        echo "No more pages to annotate";
        exit();
    }
    // If checkpoint file contains a saved state → resume from last progress
    elseif($checkPoints[0] != 'none' && $checkPoints[1] != 'none'){
        //If we are not starting for the first time (none means there is no checkpoint)
        $_SESSION['pageName'] = $checkPoints[0];
        $_SESSION['tableNumber'] = $checkPoints[1];
        $_SESSION['tableName'] = 'table' . $checkPoints[1];
    }
    // If no checkpoint exists → start from the first page
    else{
        $_SESSION['pageName'] = $folders[2];
        $_SESSION['tableNumber'] = '1';
        $_SESSION['tableName'] = 'table1';
    }

    // Assign the working directory for the annotator
    if ($_SESSION['annotator'] == 1 || $_SESSION['annotator'] == 2 || $_SESSION['annotator'] == 3)
        $_SESSION['currentDirectory'] = 'webpages/' . $_SESSION['pageName'] . '/';
    
    elseif($_SESSION['annotator'] == 4 || $_SESSION['annotator'] == 5 || $_SESSION['annotator'] == 6)
        $_SESSION['currentDirectory'] = 'webpages1/' . $_SESSION['pageName'] . '/';  

    
    
    print_r($_SESSION['annotator']);
    print_r($currentPageIndex);
    print_r($_SESSION['pageName']);
    print_r($_SESSION['tableNumber']);
    print_r($_SESSION['tableName']);
    print_r($_SESSION['currentDirectory']);
    print_r($_SESSION['annotatorFile']);

    


    header('Location: annotator.php');
    exit();
}
// If no folders found in /webpages/ → display error
else{
    echo "No folders (pages) found in the /webpages/ directory to annotate.";
}

?>