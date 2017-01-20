module Main where

data Fitness = Fitness Float Int Int Float deriving Show
data Size = Size Float Int Int Float deriving Show
data Generation = Generation Int Int Fitness Size deriving Show

convert :: String -> Generation
convert s = Generation (read gen :: Int) (read nevals :: Int) mkFit mkSize
    where 
        [gen, nevals, favg, fmax, fmin, fstd, savg, smax, smin, sstd] = words s
        mkFit = (Fitness (read favg :: Float) (read fmax :: Int) (read fmin :: Int) (read fstd :: Float))
        mkSize = (Size (read savg :: Float) (read smax :: Int) (read smin :: Int) (read sstd :: Float))

readFile' :: Int -> IO String
readFile' num = readFile $ "log-" ++ (show num :: String) ++ ".txt"

parseFile :: IO String -> IO [Generation]
parseFile f = map convert . drop 3 . lines <$> f

main :: IO ()
main = do
    stats <- mapM (parseFile . readFile') [100..125]
    print $ stats